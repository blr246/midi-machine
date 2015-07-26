--[
-- NN package model helpers.
--
-- (c) 2014 Brandon L. Reiss
--]
require "nn"
require "math"
require "torch"

require "Rleu"
require "RluMax"
require "SgdMomentum"

models = {}

---Append dataset information to a model.
function models.append_ds_info(ds, mlp)
    mlp.time_sig = ds.time_sig
    mlp.dims = ds.logical_dims
    return mlp
end

---Create a simple 2-layer NN.
--Each output channel is taken in isolation using this architecture. The
--filters in layer 2 are applied to each note channel and accumulated into the
--hidden features with num_hidden per note track. The output layer is a linear
--combination of these hidden features where each feature is still isolated to
--a single note.
function models.simple_2lnn_iso(ds, num_hidden)

    local mlp = models.append_ds_info(ds, nn.Sequential())
    local input_len = mlp.dims.input[2]
    local output_len = mlp.dims.output[2]

    -- Project to num_notes x num_hidden.
    mlp:add(nn.Linear(input_len, num_hidden))
    mlp:add(nn.Rleu())

    -- Project to num_notes x output_len.
    mlp:add(nn.Linear(num_hidden, output_len))

    mlp:add(nn.Tanh())

    return mlp
end

---Create a simple 2-layer NN.
--Each output channel is a linear combination over all hidden features for all
--note tracks. Therefore, the activation of note 1 is influenced by the
--activation of all other notes. This requires more weights than the isolated
--architecture.
function models.simple_2lnn_cmb(ds, num_hidden)

    local mlp = models.append_ds_info(ds, nn.Sequential())

    local num_notes = mlp.dims.input[1]
    local input_len = mlp.dims.input[2]
    local output_len = mlp.dims.output[2]

    -- By transposing, we extract features over the notes for each time slice.

    -- Transpose to input_len x num_notes and project to input_len x num_hidden.
    mlp:add(nn.Transpose({1, 2}))
    mlp:add(nn.Linear(num_notes, num_hidden))
    mlp:add(nn.Rleu())

    -- Transpose to num_hidden x input_len and project to num_hidden x output_len.
    mlp:add(nn.Transpose({1, 2}))
    mlp:add(nn.Linear(input_len, output_len))
    mlp:add(nn.Rleu())

    -- Transpose to output_len x num_hidden and project to output_len x num_notes.
    mlp:add(nn.Transpose({1, 2}))
    mlp:add(nn.Linear(num_hidden, num_notes))
    mlp:add(nn.Rleu())

    -- Transpose to num_notes x output_len.
    mlp:add(nn.Transpose({1, 2}))

    mlp:add(nn.Tanh())

    return mlp
end

---Create a simple 2-layer NN.
--Each output channel is a linear combination over all hidden features for all
--note tracks. Therefore, the activation of note 1 is influenced by the
--activation of all other notes. This requires more weights than the isolated
--architecture.
function models.simple_2lnn_iso_cmb(ds, num_hidden)

    local mlp = models.append_ds_info(ds, nn.Sequential())
    local input_len = mlp.dims.input[2]
    local output_len = mlp.dims.output[2]

    -- Project to num_notes x num_hidden.
    mlp:add(nn.Linear(input_len, num_hidden))
    mlp:add(nn.Rleu())

    -- Transpose to num_hidden x num_notes and project to num_hidden x num_hidden.
    mlp:add(nn.Transpose({1, 2}))
    mlp:add(nn.Linear(num_notes, num_hidden))
    mlp:add(nn.Rleu())

    -- Project to num_hidden x output_len.
    mlp:add(nn.Linear(num_hidden, output_len))
    mlp:add(nn.Rleu())

    -- Transpose to output_len x num_hidden and project to output_len x num_notes.
    mlp:add(nn.Transpose({1, 2}))
    mlp:add(nn.Linear(num_hidden, num_notes))

    mlp:add(nn.Tanh())

    return mlp
end

--- Train model using the given dataset.
function models.train_model(ds, model, criterion, train_args)

    local train = ds.data_train()
    local trainer = nn.SgdMomentum(model, criterion, train_args)
    trainer:train(train)

    local train_loss = 0
    for i = 1, train:size() do
        local X = train[i][1]
        local Y = train[i][2]
        local loss = criterion:forward(model:forward(X), Y)
        train_loss = train_loss + loss
    end
    local avg_train_loss = train_loss / train:size()

    local test_loss = 0
    local test = ds.data_test()
    for i = 1, test:size() do
        local X = test[i][1]
        local Y = test[i][2]
        local loss = criterion:forward(model:forward(X), Y)
        test_loss = test_loss + loss
    end
    local avg_test_loss = test_loss / test:size()

    return avg_train_loss, avg_test_loss
end

--- [0, 255] => [-1, 1]
function models.from_byte(x)
    if type(x) == "userdata" then
        return x:div(127.5):add(-1)
    else
        return (x / 127.5) - 1
    end
end

--- [-1, 1] => [0, 255]
function models.to_byte(x)
    if type(x) == "userdata" then
        return x:map(x, function(xx)
            return math.max(0, math.min(255, math.floor((xx + 1) * 127.5)))
        end)
    else
        return math.max(0, math.min(255, math.floor((x + 1) * 127.5)))
    end
end

---Transform data into range [-1, 1].
function models.normalize_data(ds)
    for _, source in ipairs(ds.sources) do
        models.from_byte(source.data)
    end
end

--- Predict a song by seeding with input window x0.
-- :param number length: the length of the song in output windows
function models.predict(model, x0, length)

    if x0:size(1) ~= model.dims.input[1]
        or x0:size(2) ~= model.dims.input[2] then
        error(string.format("Seed point has incorrect dims (%d, %d) != (%d, %d)",
                            x0:size(1), x0:size(2),
                            model.dims.input[1], model.dims.input[2]))
    end

    local input_wnd = model.dims.input[2]
    local output_wnd =  model.dims.output[2]

    local total_size = output_wnd * length
    local channel_dims = model.dims.input[1]
    local total_length = input_wnd + total_size
    local x0_song = torch.Tensor(channel_dims, total_length)
    local song = x0_song:narrow(2, input_wnd + 1, total_size)

    -- Copy first input to output buffer.
    x0_song:narrow(2, 1, input_wnd):copy(x0)

    for offset = 1, total_size, output_wnd do
        -- Predict next output_wnd and copy to the song. 
        local X = x0_song:narrow(2, offset, input_wnd)
        local Y = model:forward(X)
        -- Normalize the output.
        models.from_byte(models.to_byte(Y))
        song:narrow(2, offset, output_wnd):copy(Y)
        print(string.format("predict() : t=%d/%d, max_ouput=% .4f",
                            offset, total_size, Y:max()))
    end

    return models.to_byte(song)
end

return models
