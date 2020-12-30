K = 2; %Number of terms per layer
N = 2; %Number of layers
D = 2; %Dimension of data
M = 500; %Number of datapoints
X = linspace(0,pi/2,M);
x = [cos(X)+0.05*rand(1,M); sin(X)+0.1*rand(1,M)];

w0 = 2*randn(N,1);
W = 1*randn(N,K);
A = 0.1*randn(N,K);
B = 0.1*randn(N,K);
alpha = log(0.001);
dv = 1e-7; %Variable differential for estimating derivatives
eta = 1e-7;

sigma = @(x) exp(alpha)*x.*(x<0)+x.*(x>=0);

update = ['S.Ti = @(z,i,S) S.w0(i)*ones(S.D,1) +',...
        'sum(  ones(S.D,1)*exp(S.W(i,:)).*sigma(z*exp(S.A(i,:))+',...
        'ones(S.D,1)*S.B(i,:)) ,2 );',...
        'T = @(z,S) Tcomp(z,N,S);'];

% % Let the base distribution be Cauchy, because right now T inverse maps to
% % some very high values.
% MM = [0;0];
% BB = [1;0.2];
% pu = @(u) BB(1)*BB(2)/pi^2*( (u(1)-MM(1))^2+BB(1)^2 )*( (u(2)-MM(2))^2+BB(2)^2 );

MU = [0.6;0.6];
SIG = [0.2 0; 0 0.2];
pu = @(u) (2*pi)^(-D/2)*det(SIG)^(-1/2)*exp(-1/2*(u-MU)'*SIG*(u-MU));

S.K=K; S.N=N; S.D=D; S.M=M; S.w0=w0; S.W=W; S.A=A; S.B=B; S.dv=dv; S.pu=pu;

eval(update);
L1 = Loss(x,S,T);
L2 = Inf;

delta = 1e-10;
maxitr = 1e4;
numitr = 0;

while abs(L2-L1)>delta && numitr<maxitr
    ogL = L1;
    %Optimizing for each parameter matrix. Probably could be written more
    %concisely with some slick eval statements, but it wouldn't improve
    %computation by much if any, so.
    ogw0 = S.w0;
    nw0 = ogw0;
    for i=1:N
        S.w0 = ogw0 + dv*(1:N==i)';
        eval(update);
        L = Loss(x,S,T);
        nw0 = nw0 - (eta/dv)*(L-ogL)*(1:N==i)';
    end
    S.w0 = nw0;
    
    ogW = S.W;
    nW = ogW;
    for i=1:N
        for j=1:K
            S.W = ogW + dv*(1:N==i)'*(1:K==j);
            eval(update);
            L = Loss(x,S,T);
            nW = nW - (eta/dv)*(L-ogL)*(1:N==i)'*(1:K==j);
        end
    end
    S.W = nW;
    
    ogB = S.B;
    nB = ogB;
    for i=1:N
        for j=1:K
            S.B = ogB + dv*(1:N==i)'*(1:K==j);
            eval(update);
            L = Loss(x,S,T);
            nB = nB - (eta/dv)*(L-ogL)*(1:N==i)'*(1:K==j);
        end
    end
    S.B = nB;
    
    ogA = S.A;
    nA = ogA;
    for i=1:N
        for j=1:K
            S.A = ogA + dv*(1:N==i)'*(1:K==j);
            eval(update);
            L = Loss(x,S,T);
            nA = nA - (eta/dv)*(L-ogL)*(1:N==i)'*(1:K==j);
        end
    end
    S.A = nA;
    
    eval(update);
    L2 = L1;
    L1 = Loss(x,S,T);
    if mod(numitr,10)==0
        fprintf('We have advanced Loss from %d to %d\n',L2,L1);
        fprintf('                 (change is %d)\n\n'  ,L1-L2);
%         u = zeros(size(x));
%         u(1,:) = 0.2*tan(pi*(rand(1,M)-1/2));
%         u(2,:) = tan(pi*(rand(1,M)-1/2));
        u = [0.6;0.6] + sqrt(0.2)*randn(2,M);
        hold off
        scatter(u(1,:),u(2,:),'r');
        hold on
        scatter(x(1,:),x(2,:),'g');
        y = zeros(size(x));
        for i=1:M
            y(:,i) = T(u(:,i),S);
        end
        scatter(y(1,:),y(2,:),'b');
        axis([-6 14 -30 30]);
        pause(0.00001);
    end
    numitr = numitr + 1;
end


function y = Tcomp(z,i,S)
    if i==1
        y = S.Ti(z,i,S);
    else
        y = S.Ti(Tcomp(z,i-1,S),i,S);
    end
end

function x = Tinv(y,S,T)
    abstol = 1e-6;
    maxitr = 1e3;
    numitr = 0;
    x = zeros(size(y));
    for i=1:size(y,2)
        x(:,i) = 0.1*ones(S.D,1);
        while norm(T(x(:,i),S)-y(:,i))>abstol && numitr<maxitr
            x(:,i) = x(:,i) - JT(x(:,i),S,T)\(T(x(:,i),S)-y(:,i));
        end
    end
end

function X = JT(z,S,T)
    X = diag( (T(z+S.dv*ones(S.D,1),S)-T(z,S))/S.dv  );
end

function [p,d] = Process(z,S,T)
% p is the base distribution vector, d the determinant vector at [z1 z2
% ...]
    p = zeros(1,size(z,2));
    d = p;
    for i=1:size(z,2)
        p(i) = S.pu(Tinv(z(:,i),S,T));
        d(i) = prod(diag(JT(z(:,i),S,T)));
    end
end

function L = Loss(x,S,T)
    [p,d] = Process(x,S,T);
    L = -1/S.M * sum(log(p)+log(d));
end



























