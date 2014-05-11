require "nn"
require "math"

local PerceptualLoss, parent = torch.class('nn.PerceptualLoss', 'nn.Criterion')

---Takes to_byte in order to suppress any gradient when
--the resulting conversion to byte has zero loss.
function PerceptualLoss:__init(to_byte, lambda)
    parent.__init(self)
    self.t1 = torch.Tensor()
    self.sizeAverage = true

    self.lambda = self.lambda or 2

    if nil == to_byte then
        error("Must supply to_byte() function")
    end

    self.loss_func = function(_, xx, yy)
        local X = to_byte(xx)
        local Y = to_byte(yy)
        local diff = xx - yy

        if X == 0 and Y > 0 then
            return diff^2 * self.lambda
        else
            return diff^2
        end
    end

    self.grad_func = function(_, xx, yy)
        local X = to_byte(xx)
        local Y = to_byte(yy)
        local diff = xx - yy

        if X == 0 and Y > 0 then
            return diff * 2 * self.lambda
        else
            return diff * 2
        end
    end
 end

function PerceptualLoss:updateOutput(input, target)
    local X = input
    local Y = target
    local t1 = self.t1:resizeAs(input)

    self.output = t1:map2(X, Y, self.loss_func):sum()

    if self.sizeAverage then
        self.output = self.output / input:nElement()
    end

    return self.output
end

function PerceptualLoss:updateGradInput(input, target)
    local X = input
    local Y = target
    local gradInput = self.gradInput:resizeAs(input)

    gradInput:map2(X, Y, self.grad_func)

    if self.sizeAverage then
        gradInput:div(input:nElement())
    end

    return gradInput
end
