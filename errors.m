flag = 3;

if (flag==0)
    % ERROR DIM=2, LAGRANGE, FE_SPACE=1:
    errors_inf = [0.916635, 0.39873	, 0.117592,  0.0300353, 0.007337 , 0.001612];
    errors_L2 = [0.468609, 0.204076 ,0.0524503 ,0.0136206, 0.003436, 0.000885];
    errors_H1 =[4.06367,3.27098	,1.69093, 0.864633 , 0.431939 ,0.213396 ];
    errors_DG = [6.45216,4.87842, 2.27482 ,1.08379 , 0.531930 ,0.258958 ];
elseif (flag==1)
% ERROR DIM=2, LAGRANGE, FE_SPACE=2:
    errors_inf = [0.662445, 0.136187, 0.0164221,  0.00239106, 0.000470 , 0.000438];
    errors_L2 = [0.153321, 0.027537 ,0.00456782 ,0.000610755, 0.000167 , 0.000150];
    errors_H1 =[2.93477,1.00002	,0.266708, 0.0682657 , 0.017142 , 0.004582 ];
    errors_DG = [6.20694,1.93232, 0.533099 ,0.140828 , 0.034399 ,0.008064];
elseif (flag==2)
% ERROR DIM=2, LAGRANGE-BRAIN, FE_SPACE=1:
    errors_inf = [0.210249, 0.0860672, 0.0249047, 0.00367873, 0.000896 , 0.000636];
    errors_L2 = [0.279924, 0.0770018 ,0.0199309 ,0.00496913, 0.001254  , 0.000356];
    errors_H1 =[3.44953,2.26354	,1.24732, 0.635389 ,  0.326080 ,0.174836 ];
    errors_DG = [3.67976,2.51012, 1.31724 ,0.651036 , 0.333629 ,0.197239];
elseif (flag==3)
    % ERROR DIM=2, LAGRANGE-BRAIN, FE_SPACE=2:
    errors_inf = [0.350261, 0.111563, 0.00903925, 0.00075927, 0.000439  ,0.000439 ];
    errors_L2 = [0.0856513, 0.0203543 , 0.00232008 , 0.000320285, 0.000154  , 0.000149];
    errors_H1 =[1.99572,0.93024	,0.224222, 0.0542274 , 0.013240 , 0.003972 ];
    errors_DG = [3.72373,1.61929, 0.291792 ,0.0660571, 0.023693 ,0.008604];
end

for i=1:length(errors_inf)
    err_inf(i) = errors_inf(end +1 -i)
    err_L2(i) = errors_L2(end +1 -i)
    err_H1(i) = errors_H1(end+1 -i)
    err_DG(i) = errors_DG(end+1 -i)
end

h = [0.0220971,0.0441942, 0.0883883,0.176777,0.353553,0.707107] ;


figure()
subplot(1,2,1)
loglog(h,err_L2/err_L2(end),'LineWidth',2)
hold on 
grid on
loglog(h,(h/h(end)).^2, "k-")
loglog(h,(h/h(end)).^3,"k--")
loglog(h,(h/h(end)).^4,"ko-")
legend("err_{L2}", "h^2", "h^3","h^4")
title("ErrorL2")

subplot(1,2,2)
loglog(h,err_inf/err_inf(end),'LineWidth',2)
hold on 
grid on
loglog(h,(h/h(end)).^2, "k-")
loglog(h,(h/h(end)).^3,"k--")
loglog(h,(h/h(end)).^4,"ko-")
legend("err_{inf}", "h^2", "h^3","h^4")
title("ErrorInf")

figure()
subplot(1,2,1)
loglog(h,err_H1/err_H1(end),'LineWidth',2)
hold on 
grid on
loglog(h,(h/h(end)).^2, "k-")
loglog(h,(h/h(end)).^3,"k--")
loglog(h,(h/h(end)),"ko-")
legend("err_{H1}", "h^2", "h^3","h")
title("ErrorH1")

subplot(1,2,2)
loglog(h,err_DG/err_DG(end),'LineWidth',2)
hold on 
grid on
loglog(h,(h/h(end)).^2, "k-")
loglog(h,(h/h(end)).^3,"k--")
loglog(h,(h/h(end)),"ko-")
legend("err_{DG}", "h^2", "h^3","h")
title("ErrorDG")


