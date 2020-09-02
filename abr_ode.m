function dpop = abr_ode(t, pop)
ka = 400; da =2; dab = 400;
kb = 2; kbb = 30;  dba = 30;
db = 2.8; K = 0.3;
kr = 0.1; krb = 1.5; dr = 0.2;

la = pop(1);
lb = pop(2);
lr = pop(3);
ra = pop(4);
rb = pop(5);
rr = pop(6);

dpop(1,1) = ka*(1-la-ra)*lr - da*la - dab*la*lb^2;
dpop(2,1) = (1-lb-rb)*(kb + kbb*lb) - db*lb - dba*la*lb^2;
dpop(3,1) = (1-lr-rr)*(kr + krb*lb) - dr*lr;
dpop(4,1) = ka*(1-la-ra)*rr - da*ra - dab*ra*rb^2;
dpop(5,1) = (1-lb-rb)*(kb + kbb*rb) - db*rb - dba*ra*rb^2;
dpop(6,1) = (1-lr-rr)*(kr + krb*rb) - dr*rr;
end