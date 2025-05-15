function full_time = time_calc(time)

minutes = floor(time / 60);
seconds = floor(mod(time, 60));
milliseconds = round(mod(time, 1) * 1000);

full_time = sprintf('%02d:%02d.%03d', minutes, seconds, milliseconds);

end

