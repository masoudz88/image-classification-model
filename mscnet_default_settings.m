function params = mscnet_default_settings()

params.image_size = [32 32];
params.pixel_range = [-4 4];

% Network settings
params.net.initial_filter_size = 3;
params.net.num_initial_filters = 16;
params.net.initial_stride = 1;
params.net.num_labels = 10;
params.net.stack_depth= [4 3 2];
params.net.num_filters= [16 32 64];
params.net.initial_pooling_layer = "none";

% Training options
params.train.mini_batch_size = 128;
params.train.learn_rate = 0.01 * params.train.mini_batch_size / 128;
params.train.optimization_algorithm = "sgdm";
params.train.max_epochs = 20;
params.train.shuffle = "every-epoch";
params.train.plot = "training-progress";
params.train.verbose = false;
params.train.learn_rate_schedule = "piecewise";
params.train.learn_rate_drop_factor = 0.1;
params.train.learn_rate_drop_period = 60;
end
