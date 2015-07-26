require "nn"
local Rleu = torch.class('nn.Rleu', 'nn.Module')

function Rleu:__init()
    self.output = torch.Tensor()
    self.gradInput = torch.Tensor()
    self.temp = torch.Tensor()
end

-- Output log(1 + e^x).
function Rleu:updateOutput(input)
    local output = self.output:resizeAs(input)
    torch.exp(output, input):add(1.0):log()
    return output
end

-- Gradient is e^x / (1 + e^x).
function Rleu:updateGradInput(input, gradOutput)
    local gradInput = self.gradInput:resizeAs(input)
    local temp = self.temp:resizeAs(input)
    torch.exp(gradInput, input)
    temp:add(gradInput, 1.0);
    gradInput:cdiv(temp)
    gradInput:cmul(gradOutput)
    return gradInput
end
