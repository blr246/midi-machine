--[
-- NN package model helpers.
--
-- (c) 2014 Brandon L. Reiss
--]
require "nn"
require "math"
require "torch"

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
    mlp:add(nn.Linear(mlp.dims.input[2], num_hidden))
    mlp:add(nn.RluMax())
    mlp:add(nn.Linear(num_hidden, mlp.dims.output[2]))

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
    -- By transposing, we extract features over the notes for each time slice.
    mlp:add(nn.Transpose({1, 2}))
    mlp:add(nn.Linear(num_notes, num_hidden))
    mlp:add(nn.RluMax())
    -- Flatten the network.
    local flattened_len = mlp.dims.input[2] * num_hidden
    mlp:add(nn.Reshape(flattened_len))
    -- Allow output channels to take features across notes.
    mlp:add(nn.Linear(flattened_len, num_notes * mlp.dims.output[2]))
    mlp:add(nn.Reshape(num_notes, mlp.dims.output[2]))

    return mlp
end

---Create a simple 2-layer NN.
--Each output channel is a linear combination over all hidden features for all
--note tracks. Therefore, the activation of note 1 is influenced by the
--activation of all other notes. This requires more weights than the isolated
--architecture.
function models.simple_2lnn_iso_cmb(ds, num_hidden)

    local mlp = models.append_ds_info(ds, nn.Sequential())
    local num_notes = mlp.dims.input[1]
    mlp:add(nn.Linear(mlp.dims.input[2], num_hidden))
    mlp:add(nn.RluMax())
    -- Flatten the network.
    local flattened_len = num_notes * num_hidden
    mlp:add(nn.Reshape(flattened_len))
    -- Allow output channels to take features across notes.
    mlp:add(nn.Linear(flattened_len, num_notes * mlp.dims.output[2]))
    mlp:add(nn.Reshape(num_notes, mlp.dims.output[2]))

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
        song:narrow(2, offset, output_wnd):copy(Y)
        print(string.format("predict() : t=%d/%d, max_ouput=% .4f",
                            offset, total_size, Y:max()))
    end

    return song
end

return models
