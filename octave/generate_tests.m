n_states = 5
n_symbols = 17

% randfixedsum constrains the columns. therefore results have to be transposed

s = randfixedsum(n_states,1, 1, 0, 1)'

t = randfixedsum(n_states, n_states, 1,0,1)'

e = randfixedsum(n_symbols, n_states, 1, 0, 1)'


observation_sequence_length = 10
o = randi(n_symbols, 1, observation_sequence_length);

[A, B, C, LogP, Alfa, Beta, Gama, Tau, Taui, Nu, Omega] = baum_welch_norm_return_all_vars(t, e, o, s)

save ("-hdf5", "tests/test1.hdf5", "s", "t", "e", "o", "A", "B", "C", "LogP", "Alfa", "Beta", "Gama", "Tau", "Taui", "Nu", "Omega")