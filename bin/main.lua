--[
-- Main command-line application for ingesting and training JSB Chorales MIDI
-- files and then building a learning machine that attempts to produce
-- Bach-like music given a sequence of initial notes.
--
-- (c) 2014 Brandon L. Reiss
--]
lapp = require 'pl.lapp'
path = require 'pl.path'
require "os"
require "paths"

-- Get lua module path.
local bin_dir = paths.dirname(paths.thisfile())
local lua_dir = path.abspath(path.join(bin_dir, "../lua"))
local mod_pattern = path.join(lua_dir, "?", "init.lua")
local script_pattern = path.join(lua_dir, "?.lua")
package.path = mod_pattern..";"..script_pattern..";"..package.path

require 'mid'
require 'models'
require 'nn'
require 'Rnn'

--[
-- TODO:
--
--   REGRESSION
--   a) Create simple deep network predicting next notes across channels as a
--      form of regression. This simplifies dealing with softmax and class
--      labels on the output, but in general it may be more difficult to get
--      reasonable performance.
--   b) Experiment with the loss function on the regression problem.
--
--   MULTICLASS
--   a) Use K classes to represent note velocity obtained by median filtering
--      the velocity values observed in the training data. Applying softmax
--      to the output gives a simple loss function.
--
--   CONVNET
--   a) Try several deep convnet architectures.
--
--   For all models, experiment with input dimensions (window size) and also
--   output dimensions.
--
--   RECURRENT
--   a) Experiment with the unrolling length.
--]

local args = lapp [[
Train a learning machine to predict sequences of notes extracted from MIDI
files.
  -i, --input-window-size (default 10) size in gcd ticks of input point X
  -o, --output-window-size (default 1) size in gcd ticks of target point Y
  -r, --rnn is a recurrent neural network; note that output window must be > 1
  -s, --dataset-train-split (default 0.9) percentage of data to use for training
  -h, --hidden-units (default 256) number of 2lnn hidden units
  -t, --model-type (default "iso") model type in {iso, iso+cmb, cmb}
                   iso - isolated, features apply to a single note
                   cmb - combined, features apply across notes
                   iso+cmb - isolated features combined across notes
  -c, --custom-labels (string) custom labels to insert at the end of the
                      output filenames (ex "lr-1e-4-mb-5k" if we were
                      experimenting with learning rate and minibatch size)
  <INPUT_DIR> (string) directory where input *.mid files reside
  <TIME_SIG_CHANNELS_GCD> (string) time signature, channels, and gcd
                          e.g. 4/2-8-24-4-256
  <OUTPUT_DIR> (string) directory used to save output model file
               and an example generated song
]]

local ds
if args.rnn then
    ds = mid.dataset.load_rnn(
            args.INPUT_DIR,
            args.TIME_SIG_CHANNELS_GCD, 
            args.input_window_size,
            args.output_window_size,
            args.dataset_train_split
            )
else
    ds = mid.dataset.load(
            args.INPUT_DIR,
            args.TIME_SIG_CHANNELS_GCD, 
            args.input_window_size,
            args.output_window_size,
            args.dataset_train_split
            )
end

-- Show command-line options.
for key, value in pairs(args) do
    print(key, value)
end

-- Generate date string used to label model and test midi.
local date_str = os.date("%Y%m%d_%H%M%S")
print('Date string: '..date_str)
print('Num training: '..ds.data_train():size())

-- Create 2-layer NN with specified hidden units. Each hidden unit is a feature
-- extractor that is applied to an input time slice for a single note.
local model
local train_args
if "iso" == args.model_type then
    model = models.simple_2lnn_iso(ds, args.hidden_units)
    train_args = { learning_rate = 1e-2, learning_rate_decay = 0.1 }
elseif "cmb" == args.model_type then
    model = models.simple_2lnn_cmb(ds, args.hidden_units)
    train_args = { learning_rate = 1e-2, learning_rate_decay = 0.1 }
elseif "iso+cmb" == args.model_type then
    model = models.simple_2lnn_iso_cmb(ds, args.hidden_units)
    train_args = { learning_rate = 1e-2, learning_rate_decay = 0.3 }
end


-- Create RNN when requested.
local train_model
local model_type
if args.rnn then
    train_model = nn.Rnn(model)
    model_type = "rnn-"..args.model_type
else
    train_model = model
    model_type = "2lnn-"..args.model_type
end

local criterion = nn.MSECriterion()

-- Initialize weights.
local stdv = 1.0
local initialized = false
while not initialized do

    local num_good = 0
    local data = ds.data_train()
    for i = 1, data:size() do
        model:reset(stdv)
        local loss = criterion:forward(train_model:forward(data[i][1]), data[i][2])
        if num_good > 100 then
            initialized = true
            break
        elseif loss == loss
            and loss < data[i][2]:nElement() then
            num_good = num_good + 1
        else
            stdv = stdv / 2
            num_good = 0
        end
    end
end
print("Initialized weights stdv="..stdv)

-- Set max iterations based on points in the dataset.
train_args.max_iteration = math.ceil(1e5 / (args.output_window_size * ds.num_train))

-- Get point that we will use to generate a song.
local gen_x0 = ds.data_test()[1][1]:narrow(2, 1, args.input_window_size)

-- Train.
local err_train, err_test = models.train_model(ds, train_model, criterion, train_args)
print("avg error train/test", err_train, err_test)

-- Append command-line options to the model.
for key, value in pairs(args) do
    model[key] = value
end

-- Write out the model.
local model_filename = 'model-'..model_type..args.hidden_units..'-'..date_str
if args.custom_labels ~= nil then
    model_filename = model_filename.."-"..args.custom_labels
end
local model_output_path = path.join(args.OUTPUT_DIR, model_filename)
torch.save(model_output_path, model)
print("Wrote model "..model_output_path)

-- Generate a song.
local song_data = models.predict(model, gen_x0, 10)
local song = mid.dataset.compose(ds.sources[1], song_data, 4)

-- Write out generated song.
local gen_filename = 'gen-'..date_str..'.mid'
local gen_output_path = path.join(args.OUTPUT_DIR, gen_filename)
mid.data.write(song.middata, io.open(gen_output_path, 'w'))
print("Wrote song "..gen_output_path)
