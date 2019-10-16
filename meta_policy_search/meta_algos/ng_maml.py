from meta_policy_search.utils import logger
from meta_policy_search.meta_algos.base import MAMLAlgo
from meta_policy_search.optimizers.maml_first_order_optimizer import MAMLPPOOptimizer
from meta_policy_search.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, _flatten_params, _unflatten_params, FiniteDifferenceHvp
from meta_policy_search.utils import *

import tensorflow as tf
import numpy as np
from collections import OrderedDict

class NG_MAML(MAMLAlgo):
    """
    ProMP Algorithm

    Args:
        policy (Policy): policy object
        name (str): tf variable scope
        learning_rate (float): learning rate for optimizing the meta-objective
        num_ppo_steps (int): number of ProMP steps (without re-sampling)
        num_minibatches (int): number of minibatches for computing the ppo gradient steps
        clip_eps (float): PPO clip range
        target_inner_step (float) : target inner kl divergence, used only when adaptive_inner_kl_penalty is true
        init_inner_kl_penalty (float) : initial penalty for inner kl
        adaptive_inner_kl_penalty (bool): whether to used a fixed or adaptive kl penalty on inner gradient update
        anneal_factor (float) : multiplicative factor for annealing clip_eps. If anneal_factor < 1, clip_eps <- anneal_factor * clip_eps at each iteration
        inner_lr (float) : gradient step size used for inner step
        meta_batch_size (int): number of meta-learning tasks
        num_inner_grad_steps (int) : number of gradient updates taken per maml iteration
        trainable_inner_step_size (boolean): whether make the inner step size a trainable variable

    """
    def __init__(
            self,
            *args,
            name="ng_maml",
            learning_rate=1e-3,
            num_minibatches=1,
            inner_lr=0.01,
            step_size=0.01,
            init_inner_kl_penalty=1e-2,
            adaptive_inner_kl_penalty=True,
            anneal_factor=1.0,
            **kwargs
            ):
        super(NG_MAML, self).__init__(*args, **kwargs)

        self.inner_lr = inner_lr
        self.step_size = step_size
        self.adaptive_inner_kl_penalty = adaptive_inner_kl_penalty
        self.inner_kl_coeff = init_inner_kl_penalty * np.ones(self.num_inner_grad_steps)
        self.anneal_coeff = 1
        self.anneal_factor = anneal_factor
        self._optimization_keys = ['observations', 'actions', 'advantages', 'agent_infos']
        self.name = name
        self.kl_coeff = [init_inner_kl_penalty] * self.meta_batch_size * self.num_inner_grad_steps

        self.build_graph()

    def _adapt_objective_sym(self, action_sym, adv_sym, dist_info_old_sym, dist_info_new_sym):
        with tf.variable_scope("likelihood_ratio"):
            likelihood_ratio_adapt = self.policy.distribution.likelihood_ratio_sym(action_sym,
                                                                                   dist_info_old_sym, dist_info_new_sym)
        with tf.variable_scope("surrogate_loss"):
            surr_obj_adapt = -tf.reduce_mean(likelihood_ratio_adapt * adv_sym)
        return surr_obj_adapt

    def build_graph(self):
        """
        Creates the computation graph
        """

        """ Create Variables """
        with tf.variable_scope(self.name):
            self.step_sizes = self._create_step_size_vars()

            # self.meta_op_phs_dict = OrderedDict()
            obs_phs, action_phs, adv_phs, dist_info_old_phs, all_phs_dict = self._ng_make_input_placeholders('')
            # self.adapt_input_ph_dict = all_phs_dict # todo original code
            self.adapt_input_ph_dict = OrderedDict()
            for d in all_phs_dict:
                self.adapt_input_ph_dict = {**self.adapt_input_ph_dict, **d}
            # self.meta_op_phs_dict.update(all_phs_dict)
            distribution_info_vars, current_policy_params = [], []
            # all_surr_objs, all_inner_kls = [], []

        for i in range(self.meta_batch_size):
            dist_info_sym = self.policy.distribution_info_sym(obs_phs[i], params=None)
            distribution_info_vars.append(dist_info_sym)  # step 0
            current_policy_params.append(self.policy.policy_params) # set to real policy_params (tf.Variable)

        with tf.variable_scope(self.name):
            """ Inner updates"""
            surr_objs, kls, adapted_policy_params = [], [], []
            self.grads, self.cgs = [], []

            # inner adaptation step for each task
            for i in range(self.meta_batch_size):
                surr_loss = self._adapt_objective_sym(action_phs[i], adv_phs[i], dist_info_old_phs[i], distribution_info_vars[i])
                kl_loss = tf.reduce_mean(self.policy.distribution.kl_sym(dist_info_old_phs[i], distribution_info_vars[i]))
                kls.append(kl_loss)
                surr_objs.append(surr_loss)
                fd = FiniteDifferenceHvp()
                cg = ConjugateGradientOptimizer(hvp_approach=fd)
                cg.build_graph(surr_loss, self.policy, all_phs_dict[i], (kl_loss, self.inner_lr))
                self.cgs.append(cg)

            """ Outer updates """ """ Objective J_LR(theta) """
            for i in range(self.meta_batch_size):
                update_param_keys = list(current_policy_params[i].keys())
                grad_obj = tf.gradients(surr_objs[i], [current_policy_params[i][key] for key in update_param_keys])
                grad_obj = tf.concat([tf.reshape(g, [-1]) for g in grad_obj], axis=0)
                # gradient = dict(zip(update_param_keys, grad))
                self.grads.append(grad_obj)



            self.ph_original_directions = []
            self.ph_adapted_directions = []
            self.fisher_vector_products = []
            self.jacobian_vector_products = []
            self.hessian_obj_vector_products = []
            for i in range(self.meta_batch_size):
                update_param_keys = list(current_policy_params[i].keys())
                grad_kl = tf.gradients(kls[i], [current_policy_params[i][key] for key in update_param_keys])
                grad_kl = tf.concat([tf.reshape(grad, [-1]) for grad in grad_kl], axis=0)
                ph = tf.placeholder(dtype=tf.float32, shape=[None], name='task_%d_original_direction'%i)
                self.ph_original_directions.append(ph)
                grad_kl_product = tf.reduce_sum(grad_kl * ph)
                """ Fu """
                fisher_vector_product = tf.gradients(grad_kl_product, [current_policy_params[i][key] for key in update_param_keys])
                fisher_vector_product = tf.concat([tf.reshape(grad, [-1]) for grad in fisher_vector_product], axis=0)
                """ grad (Fu)^T * v"""
                ph_ = tf.placeholder(dtype=tf.float32, shape=[None], name='task_%d_adapted_direction'%i)
                self.ph_adapted_directions.append(ph_)
                fisher_uv = tf.reduce_sum(fisher_vector_product * ph_)
                jacobian_vector_product = tf.gradients(fisher_uv, [current_policy_params[i][key] for key in update_param_keys])
                jacobian_vector_product = tf.concat([tf.reshape(grad, [-1]) for grad in jacobian_vector_product], axis=0)
                self.fisher_vector_products.append(fisher_vector_product)
                self.jacobian_vector_products.append(jacobian_vector_product)

                """ Hessian objective * v"""
                grad_vector_product = tf.reduce_sum(self.grads[i] * ph_)
                hessian_obj_vector_product = tf.gradients(grad_vector_product, [current_policy_params[i][key] for key in update_param_keys])
                hessian_obj_vector_product = tf.concat([tf.reshape(grad, [-1]) for grad in hessian_obj_vector_product], axis=0)
                self.hessian_obj_vector_products.append(hessian_obj_vector_product)
    def _adapt(self, samples):
        """
        Performs MAML inner step for each task and stores the updated parameters in the policy

        Args:
            samples (list) : list of dicts of samples (each is a dict) split by meta task

        """
        assert len(samples) == self.meta_batch_size
        assert [sample_dict.keys() for sample_dict in samples]
        sess = tf.get_default_session()

        # prepare feed dict
        self.input_dict_list = self._ng_extract_input_dict(samples, self._optimization_keys, prefix='') # list
        input_ph_dict = self.adapt_input_ph_dict # list

        self.descent_directions = []
        adapted_policies_params = []
        policy_params = self.policy.get_param_values()
        policy_params_val = _flatten_params(policy_params)

        self.inner_lrs=[]
        for i in range(self.meta_batch_size):
            # feed_dict = utils.create_feed_dict(placeholder_dict=input_ph_dict[i], value_dict=input_dict[i])
            # compute the post-update / adapted policy parameters
            descent_direction = self.cgs[i].descent_direction(self.input_dict_list[i])
            self.descent_directions.append(descent_direction)
            adapted_policies_param = policy_params_val - descent_direction * self.cgs[i].init_step_size# todo min(self.inner_lr, self.cgs[i].init_step_size)
            self.inner_lrs.append(self.cgs[i].init_step_size)
            adapted_policies_param = _unflatten_params(adapted_policies_param, params_example=policy_params)
            adapted_policies_params.append(adapted_policies_param)
            # store the new parameter values in the policy

        self.policy.update_task_parameters(adapted_policies_params)

    def optimize_policy(self, all_samples_data, log=True):
        """
        Performs NG-MAML outer step

        Args:
            all_samples_data (list) : list of lists of lists of samples (each is a dict) split by gradient update and
             meta task
            log (bool) : whether to log statistics

        Returns:
            None
        """

        """ Calculate grad L(theta) """

        sess = tf.get_default_session()
        samples = all_samples_data[-1]
        assert len(samples) == self.meta_batch_size
        """ Calculate the value of grad J(adapted theta) """
        input_dict = self._extract_input_dict(samples, self._optimization_keys, prefix='')

        feed_dict_inputs = utils.create_feed_dict(placeholder_dict=self.adapt_input_ph_dict, value_dict=input_dict)
        feed_dict_params = self.policy.policies_params_feed_dict

        feed_dict = {**feed_dict_inputs, **feed_dict_params}
        grad_adapted_vals = sess.run(self.grads, feed_dict=feed_dict)

        """ 1. Compute F^-1 * grad(theta'), denote it as v """
        adapted_directions = []
        for i in range(self.meta_batch_size):
            adapted_direction = self.cgs[i].descent_direction_given_grad(self.input_dict_list[i], grad_adapted_vals[i])
            adapted_directions.append(adapted_direction)

        """ 2. Compute J_{Fu}^T * v """
        samples0 = all_samples_data[0]
        input_dict0 = self._extract_input_dict(samples0, self._optimization_keys, prefix='')
        feed_dict_inputs0 = utils.create_feed_dict(placeholder_dict=self.adapt_input_ph_dict, value_dict=input_dict0)
        feed_dict_params0 = self.policy.policies_params_feed_dict_pre

        feed_dict = {**feed_dict_inputs0, **feed_dict_params0}
        for i in range(self.meta_batch_size):
            feed_dict[self.ph_original_directions[i]] = self.descent_directions[i] #np.concatenate([grad.reshape(-1) for grad in grad_adapted_vals[i]])
            feed_dict[self.ph_adapted_directions[i]] = adapted_directions[i]
        jacobian_vector_products = sess.run(self.jacobian_vector_products, feed_dict=feed_dict)
        hessian_obj_vector_products = sess.run(self.hessian_obj_vector_products, feed_dict=feed_dict)

        grad_losses = []
        for i in range(self.meta_batch_size):
            loss = grad_adapted_vals[i] + self.inner_lrs[i] * jacobian_vector_products[i] - self.inner_lrs[i] * hessian_obj_vector_products[i]
            grad_losses.append(loss)

        policy_params = self.policy.get_param_values()
        policy_params_vals = _flatten_params(policy_params)
        adapted_loss = np.mean(grad_losses, axis=0)
        adapted_policy_params_val = policy_params_vals - self.step_size * adapted_loss
        adapted_policy_params_val = _unflatten_params(adapted_policy_params_val, params_example=policy_params)
        self.policy.set_params(adapted_policy_params_val)
        # self.cgs[0].gradient()

        # loss_before = self.optimizer.optimize(input_val_dict=meta_op_input_dict)



    def adapt_kl_coeff(self, kl_coeff, kl_values, kl_target):
        if hasattr(kl_values, '__iter__'):
            assert len(kl_coeff) == len(kl_values)
            return np.array([_adapt_kl_coeff(kl_coeff[i], kl, kl_target) for i, kl in enumerate(kl_values)])
        else:
            return _adapt_kl_coeff(kl_coeff, kl_values, kl_target)

def _adapt_kl_coeff(kl_coeff, kl, kl_target):
    if kl < kl_target / 1.5:
        kl_coeff /= 2

    elif kl > kl_target * 1.5:
        kl_coeff *= 2
    return kl_coeff