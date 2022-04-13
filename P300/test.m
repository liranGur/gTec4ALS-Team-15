function test()
%TEST Summary of this function goes here
%   Detailed explanation goes here


    if 1 == 2
        func = @(a,b) t1(a,b, 9);
    else
        func = @(a,b) t2(a,b);
    end

    x = func(1,3);
    display('==== ')
    display(x);
end


function [z] = t1(a,b, c)
    display('1');
    z = a + b + c;
end

function [z] = t2(a,b)
    display('2');
    z = b - a;
end