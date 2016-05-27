import lasagne.layers as L
import lasagne.nonlinearities as NL
import lasagne.init as LI
import theano


class DeterministicMLPPolicy():
    def __init__(self, env_spec, hidden_sizes=(32, 32), hidden_nonlinearity=NL.rectify, hidden_W_init=LI.HeUniform(),
                 hidden_b_init=LI.Constant(0.), output_nonlinearity=NL.tanh, output_W_init=LI.Uniform(-3e-3, 3e-3),
                 output_b_init=LI.Uniform(-3e-3, 3e-3)):

        l_obs = L.InputLayer(shape=(None, env_spec.observation_space.flat_dim))

        l_hidden = l_obs

        for idx, size in enumerate(hidden_sizes):
            l_hidden = L.DenseLayer(l_hidden, num_units=size, W=hidden_W_init, b=hidden_b_init,
                                    nonlinearity=hidden_nonlinearity, name="h%d" % idx)

        l_output = L.DenseLayer(l_hidden, num_units=env_spec.action_space.flat_dim, W=output_W_init,
                                b=output_b_init, nonlinearity=output_nonlinearity, name="output")

        # Note the deterministic=True argument. It makes sure that when getting
        # actions from single observations, we do not update params in the
        # batch normalization layers

        action_var = L.get_output(l_output, deterministic=True)
        self._output_layer = l_output

        self._f_actions = theano.function(inputs=[l_obs.input_var], outputs=action_var,
                                          on_unused_input='ignore', allow_input_downcast=True)

    def get_action(self, observation):
        action = self._f_actions([observation])[0]
        return action, dict()

    def get_actions(self, observations):
        return self._f_actions(observations), dict()

    def get_action_sym(self, obs_var):
        return L.get_output(self._output_layer, obs_var)
