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
require 'PerceptualLoss'

---Generate model filename.
local make_model_filename = function(args, date_str)
    local fn = 'model'
    if args.rnn then
        fn = fn.."-rnn-"
    else
        fn = fn.."-2lnn"
    end
    fn = fn.."-"..args.model_type
    if #args.with_reg > 0 then
        fn = fn.."-reg"..args.with_reg
    else
        fn = fn.."-noreg"
    end
    fn = fn.."-"..args.hidden_units..'-'..date_str
    if #args.custom_labels > 0 then
        fn = fn.."-"..args.custom_labels
    end
    return fn
end

---Compute weight initialization by searching for a value that does not give
--nan, inf, or very large losses compared to the number of output dims.
local find_stdv_init_weight = function(data, model, criterion)
    local stdv = 1.0
    local initialized = false
    local num_good = 0

    while not initialized do
        for i = 1, data:size() do
            model:reset(stdv)
            local loss = criterion:forward(model:forward(data[i][1]), data[i][2])
            if num_good > 100 then
                initialized = true
                break
            elseif loss == loss and loss < data[i][2]:nElement() then
                num_good = num_good + 1
            else
                stdv = stdv / 2
                num_good = 0
            end
        end
    end
    return stdv
end

--- L2 regularizer.
local l2reg = function(w, gw)
    local loss = torch.pow(gw, w, 2):sum()
    torch.mul(gw, w, 2)
    return loss
end

--- L1 regularizer.
local l1reg = function(w, gw)
    local loss = torch.abs(gw, w):sum()
    torch.sign(gw, w)
    return loss
end

local args = lapp [[
Train a learning machine to predict sequences of notes extracted from MIDI
files.
  -i, --input-window-size (default 10) size in gcd ticks of input point X
  -o, --output-window-size (default 1) size in gcd ticks of target point Y
  -r, --rnn use Recurrent Neural Network; note that output window must be > 1
  -s, --dataset-train-split (default 0.2) percentage of data to use for training;
                            *not all models require a lot of data*
  -h, --hidden-units (default 32) number of 2lnn hidden units
  -t, --model-type (default "iso") model type in {iso, iso+cmb, cmb}
                   iso - isolated, features apply to a single note
                   cmb - combined, features apply across notes
                   iso+cmb - isolated features combined across notes
  -c, --custom-labels (default "") custom labels to insert at the end of the
                      output filenames (ex "lr-1e-4-mb-5k" if we were
                      experimenting with learning rate and minibatch size)
  -w, --with-reg (default none) use regularization in {none, l1, l2}
  -d, --dont-test do not compute test set loss (useful for rapid training)
  <INPUT_DIR> (string) directory where input *.mid files reside
  <TIME_SIG_CHANNELS_GCD> (string) time signature, channels, and gcd
                          e.g. 4/2-8-24-4-256
  <OUTPUT_DIR> (string) directory used to save output model file
               and an example generated song
]]

-- Create dataset depending on RNN.
local ds_func = args.rnn and mid.dataset.load_rnn or mid.dataset.load
local ds = ds_func(args.INPUT_DIR,
                   args.TIME_SIG_CHANNELS_GCD, 
                   args.input_window_size,
                   args.output_window_size,
                   args.dataset_train_split)

-- Show command-line options.
for key, value in pairs(args) do
    print(key, value)
end

-- Generate date string used to label model and test midi.
local date_str = os.date("%Y%m%d_%H%M%S")
print('Date string: '..date_str)
print('Num training: '..ds.data_train():size())

-- Model type requested. Use some default settings for the trainer.
local model, train_args
if "iso" == args.model_type then
    model = models.simple_2lnn_iso(ds, args.hidden_units)
    train_args = { learning_rate = 1e-2, mini_batch_size = 5000, learning_rate_decay = 0.1 }
elseif "cmb" == args.model_type then
    model = models.simple_2lnn_cmb(ds, args.hidden_units)
    local lr = 1 / ds.num_train
    train_args = { learning_rate = lr, mini_batch_size = 500, learning_rate_decay = 0.1 }
elseif "iso+cmb" == args.model_type then
    model = models.simple_2lnn_iso_cmb(ds, args.hidden_units)
    local lr = 1 / ds.num_train
    train_args = { learning_rate = lr, mini_batch_size = 500, learning_rate_decay = 0.1 }
else
    error("Error: unknown model type "..args.model_type)
end

-- Create RNN when requested.
local train_model = args.rnn and nn.Rnn(model) or model

-- Select criterion and initialize weights.
--local to_byte = mid.dataset.default_to_byte
--local criterion = nn.PerceptualLoss(to_byte)
local criterion = nn.MSECriterion()
local stdv = find_stdv_init_weight(ds.data_train(), train_model, criterion)
train_model:reset(stdv)
print("reset() weights with stdv="..stdv)

-- Set max iterations based on points in the dataset.
train_args.max_iteration = math.ceil(1e6 / (args.output_window_size * ds.num_train))
print("Setting max_iteration="..train_args.max_iteration.." using heuristic; tweak as necessary")

-- Get point that we will use to generate a song.
local gen_x0 = ds.data_test()[1][1]:narrow(2, 1, args.input_window_size)

-- See if we are not going to test.
if args.dont_test then
    ds.data_test = function() return { size = function() return 0 end } end
    ds.num_test = 0
    print("Info: clearing test set; test loss will be nan")
end

-- Get regularizer when requested.
if args.with_reg == "l1" then
    train_args.reg = l1reg
elseif args.with_reg == "l2" then
    train_args.reg = l2reg
elseif args.with_reg ~= "none" then
    error("Error: unknown regularization type "..args.with_reg)
end

-- Train.
local err_train, err_test = models.train_model(ds, train_model, criterion, train_args)
print("avg error train/test", err_train, err_test)

-- Append command-line options to the model.
for key, value in pairs(args) do
    model[key] = value
end

-- Write out the model.
local model_filename = make_model_filename(args, date_str)
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
