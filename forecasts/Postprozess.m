% raw forecast

info = h5info('forecasts_5min_BNI.h5');
columns = h5read('forecasts_5min_GHI.h5', '/df/axis0');
Index = h5read('forecasts_5min_GHI.h5', '/df/axis1');
Pdc = h5read('forecasts_5min_GHI.h5', '/df/block0_values');

%% Postprocess

info = h5info('results_BNI_intra-hour.h5');
columns = h5read('results_BNI_intra-hour.h5', '/df/axis0');
Index = h5read('results_BNI_intra-hour.h5', '/df/axis1');
Werte = h5read('results_BNI_intra-hour.h5', '/df/block0_values');
dk = h5read('results_BNI_intra-hour.h5', '/df/block1_values');
error = h5read('results_BNI_intra-hour.h5', '/df/block0_items');
dataset_hor_model = h5read('results_BNI_intra-hour.h5', '/df/block1_items');