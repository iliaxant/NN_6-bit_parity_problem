function [inputs, targets] = create_dataset(digits)

decimals = 0: 2^digits - 1;

binary_strings = dec2bin(decimals, digits);
inputs = (binary_strings - '0')';

ones_sum = sum(inputs, 1);
targets = double(mod(ones_sum, 2) == 0);

end

