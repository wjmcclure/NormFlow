N = 1e4;
Loss2 = -Inf;
pu = @(x) 1/sqrt(0.2*pi)*exp(-x.^2/0.2); %base distribution
Tinv = @(x,a,b) exp(-a)*(x-b); %Transformation inverse
logdet = @(x,a,b) -ones(size(x))*a; %log Jacobian
mu = 5*randn;
sigma = 2*rand;
a = rand;
b = rand;
rng(23564623);
%DataRaw = mu + sigma*randn(N,1);
DataRaw = rand+rand(N,1);
x = DataRaw;

Loss1 = -1/N*( sum(log(pu(Tinv(x,a,b))) + logdet(x,a,b)) );

da = 1e-5; 
db = da;
eta = da;

itrcount = 0;
maxitr = 1e6;
delta = 1e-8;

updatemode = 1;
while abs(Loss1-Loss2) > delta && itrcount < maxitr
    if mod(itrcount,1000) == 0
        fprintf('             mu: %d\n',mu);
        fprintf('Current guess is\n');
        fprintf('              a: %d\n',a);
        fprintf('              b: %d\n',b);
        fprintf('Current loss is: %d\n',Loss1);
        fprintf('    And diff is: %d\n\n',abs(Loss1-Loss2));
        hold off
        histogram(DataRaw);
        u = sqrt(0.1)*randn(N,1);
        hold on
        histogram(u);
        histogram(exp(a)*u+b);
        axis([-2 5 0 800]);
        pause(0.01);
    end
    switch updatemode
        case 1 %All at once
            pu0 = log(pu(Tinv(x,a,b)));
            dpuda = 1/N*sum((log(pu(Tinv(x,a+da,b))) - pu0)/da);
            dpudb = 1/N*sum((log(pu(Tinv(x,a,b+db))) - pu0)/db);
            det0 = logdet(x,a,b);
            ddetda = 1/N*sum((logdet(x,a+da,b) - det0)/da);
            ddetdb = 1/N*sum((logdet(x,a,b+db) - det0)/db);
        case 2 %SGD
            x = DataRaw(randperm(N));
            for i=1:N
                pu0 = log(pu(Tinv(x(i),a,b)));
                dpuda = (log(pu(Tinv(x(i),a+da,b))) - pu0)/da;
                dpudb = (log(pu(Tinv(x(i),a,b+db))) - pu0)/db;
                det0 = logdet(x(i),a,b);
                ddetda = (logdet(x(i),a+da,b) - det0)/da;
                ddetdb = (logdet(x(i),a,b+db) - det0)/db;
                if any(isnan([dpuda,dpudb,ddetda,ddetdb]))
                    fprintf('WARNING!\n');
                end
            end
        case 3 %Batch SGD
            x = DataRaw(randperm(N));
            for i=1:10
                X = x((1:N/10)+(i-1)*N/10);
                pu0 = log(pu(Tinv(X,a,b)));
                dpuda = 10/N*sum((log(pu(Tinv(X,a+da,b))) - pu0)/da);
                dpudb = 10/N*sum((log(pu(Tinv(X,a,b+db))) - pu0)/db);
                det0 = logdet(X,a,b);
                ddetda = 10/N*sum((logdet(X,a+da,b) - det0)/da);
                ddetdb = 10/N*sum((logdet(X,a,b+db) - det0)/db);
            end
    end
    a = a + eta*(dpuda+ddetda);
    b = b + eta*(dpudb+ddetdb);
    Loss2 = Loss1;
    Loss1 = -1/N*( sum(log(pu(Tinv(x,a,b))) + logdet(x,a,b)) );
    itrcount = itrcount + 1;
end    
