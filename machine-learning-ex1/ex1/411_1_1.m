x = 1:10000;
y = log2(x);
z = .6*log2(.4*x)+(.4)*(log2(.4*x)+log2(.6*x));
plot(x,y,"-;log(n);",x,z,"-;.6log(.4n)+.4[log(.6n)+log(.4n)];");
xlabel("n");


y = x;
z = .6*.4*x+(.4)*(.4*x+.6*x);
plot(x,y,"-;log(n);",x,z,"-;.6log(.4n)+.4[log(.6n)+log(.4n)];");