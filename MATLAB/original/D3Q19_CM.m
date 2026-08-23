
clear all
clc
syms U V W R omega k5_star k6_star k7_star k8_star k9_star real

%% Define useful quantities
central_moments = true;
ortho = true;
% Relaxation matrix
% omega must relax ALL FIVE independent deviatoric stress moments
% (2 traceless-normal + 3 shear), i.e. positions 6..10 in this basis.
% Relaxing only 8..10 leaves the traceless-normal parts at rate 1 and makes
% the shear viscosity direction-dependent.
L = diag([1 1 1 1 1 omega omega omega omega omega 1 1 1 1 1 1 1 1 1]);
cx = [ 0,  1, -1,  0,  0,  0,  0,  1, -1,  1, -1,  1, -1,  1, -1,  0,  0,  0,  0 ];
cy = [ 0,  0,  0,  1, -1,  0,  0,  1,  1, -1, -1,  0,  0,  0,  0,  1, -1,  1, -1 ];
cz = [ 0,  0,  0,  0,  0,  1, -1,  0,  0,  0,  0,  1,  1, -1, -1,  1,  1, -1, -1 ];
w  = [ 1/3, ...
       1/18,1/18,1/18,1/18,1/18,1/18, ...
       1/36,1/36,1/36,1/36,1/36,1/36,1/36,1/36,1/36,1/36,1/36,1/36 ];
cs = 1/sqrt(3);
cs = 1/sqrt(3);
cs2 = cs^2;
cs3 = cs^3;
cs4 = cs^4;
cs5 = cs^5;
cs6 = cs^6;
cs8 = cs^8;
cs10 = cs^10;
cs12 = cs^12;
T = sym(zeros(19,19));
M = sym(zeros(19,19));
N = sym(zeros(19,19));
feq = sym(zeros(19,1));

U2 = U*U;
V2 = V*V;
W2 = W*W;
u2 = U*U+V*V+W*W;
u = [U, V, W];
for i=1:length(cx)
    % build equilibrium populations
    if(cx(i)==0 && cy(i)==0 && cz(i)==0)
        feq(i) = R/3*(1-(U2+V2+W2)+3*(U2*V2+U2*W2+V2*W2));
    elseif(cx(i)~=0 && cy(i)==0 && cz(i)==0)
        feq(i) = R/18*(1+3*cx(i)*U+3*(U2-V2-W2)-9*cx(i)*(U*V2+U*W2)-9*(U2*V2+U2*W2));
    elseif(cx(i)==0 && cy(i)~=0 && cz(i)==0)
        feq(i) = R/18*(1+3*cy(i)*V+3*(-U2+V2-W2)-9*cy(i)*(U2*V+V*W2)-9*(U2*V2+V2*W2));
    elseif(cx(i)==0 && cy(i)==0 && cz(i)~=0)
        feq(i) = R/18*(1+3*cz(i)*W+3*(-U2-V2+W2)-9*cz(i)*(U2*W+V2*W)-9*(U2*W2+V2*W2));
    elseif(cx(i)~=0 && cy(i)~=0 && cz(i)==0)
        feq(i) = R/36*(1+3*(cx(i)*U+cy(i)*V)+3*(U2+V2)+9*cx(i)*cy(i)*U*V+...
                 9*(cy(i)*U2*V+cx(i)*U*V2)+9*U2*V2);
    elseif(cx(i)~=0 && cy(i)==0 && cz(i)~=0)
        feq(i) = R/36*(1+3*(cx(i)*U+cz(i)*W)+3*(U2+W2)+9*cx(i)*cz(i)*U*W+...
                 9*(cz(i)*U2*W+cx(i)*U*W2)+9*U2*W2);
    elseif(cx(i)==0 && cy(i)~=0 && cz(i)~=0)
        feq(i) = R/36*(1+3*(cy(i)*V+cz(i)*W)+3*(V2+W2)+9*cy(i)*cz(i)*V*W+...
                 9*(cz(i)*V2*W+cy(i)*V*W2)+9*V2*W2);
    end
    
  cxi = cx(i);  cyi = cy(i);  czi = cz(i);
    cx2 = cxi*cxi;  cy2 = cyi*cyi;  cz2 = czi*czi;

    % number of non-zero components (0: rest, 1: axis, 2: face diagonal)
    nnz = cx2 + cy2 + cz2;

    if nnz == 0
        % ---- rest (0,0,0) ----
        feq(i) = R*( ...
              U^4/2 + U^2*V^2 + U^2*W^2 + U^2/6 ...
            + V^4/2 + V^2*W^2 + V^2/6 ...
            + W^4/2 + W^2/6 ...
            + 1/3 );

    elseif nnz == 1
        % ---- axis (±1,0,0) or (0,±1,0) or (0,0,±1) ----
        if cx2 == 1
            s = cxi; % s = ±1
            feq(i) = -R*( ...
                -U^4/2 + (s)*U^3/2 + (U^2*V^2)/2 + (U^2*W^2)/2 + U^2/3 ...
                + (s)*(U*V^2)/2 + (s)*(U*W^2)/2 - (s)*U/6 ...
                + V^4/2 + V^2/6 + W^4/2 + W^2/6 - 1/18 );

        elseif cy2 == 1
            s = cyi; % s = ±1
            feq(i) = -R*( ...
                U^4/2 + (U^2*V^2)/2 + (s)*(U^2*V)/2 + U^2/6 ...
              - V^4/2 + (s)*V^3/2 + (V^2*W^2)/2 + V^2/3 ...
              + (s)*(V*W^2)/2 - (s)*V/6 ...
              + W^4/2 + W^2/6 - 1/18 );

        else % cz2 == 1
            s = czi; % s = ±1
            feq(i) = -R*( ...
                U^4/2 + (U^2*W^2)/2 + (s)*(U^2*W)/2 + U^2/6 ...
              + V^4/2 + (V^2*W^2)/2 + (s)*(V^2*W)/2 + V^2/6 ...
              - W^4/2 + (s)*W^3/2 + W^2/3 - (s)*W/6 - 1/18 );
        end

    else
        % ---- face diagonals (two nonzero components, one zero) ----
        if cz2 == 0
            % (±1,±1,0)  -> signs s1=cx, s2=cy, W is the "zero" component
            s1 = cxi;  s2 = cyi;
            feq(i) = R*( ...
                -U^4/8 + (s1)*U^3/8 + (U^2*V^2)/4 + (s2)*(U^2*V)/4 + (5*U^2)/24 ...
                + (s1)*(U*V^2)/4 + (s1*s2)*(U*V)/4 + (s1)*U/12 ...
                -V^4/8 + (s2)*V^3/8 + (5*V^2)/24 + (s2)*V/12 ...
                + (3*W^4)/8 - W^2/8 + 1/36 );

        elseif cy2 == 0
            % (±1,0,±1)  -> signs s1=cx, s3=cz, V is the "zero" component
            s1 = cxi;  s3 = czi;
            feq(i) = R*( ...
                -U^4/8 + (s1)*U^3/8 + (U^2*W^2)/4 + (s3)*(U^2*W)/4 + (5*U^2)/24 ...
                + (s1)*(U*W^2)/4 + (s1*s3)*(U*W)/4 + (s1)*U/12 ...
                + (3*V^4)/8 - V^2/8 ...
                -W^4/8 + (s3)*W^3/8 + (5*W^2)/24 + (s3)*W/12 ...
                + 1/36 );

        else
            % (0,±1,±1)  -> signs s2=cy, s3=cz, U is the "zero" component
            s2 = cyi;  s3 = czi;
            feq(i) = R*( ...
                + (3*U^4)/8 - U^2/8 ...
                -V^4/8 + (s2)*V^3/8 + (V^2*W^2)/4 + (s3)*(V^2*W)/4 + (5*V^2)/24 ...
                + (s2)*(V*W^2)/4 + (s2*s3)*(V*W)/4 + (s2)*V/12 ...
                -W^4/8 + (s3)*W^3/8 + (5*W^2)/24 + (s3)*W/12 ...
                + 1/36 );
        end
    end

    % Set the basis
    if(ortho)
    CX = cx(i)-U;
        CY = cy(i)-V;
        CZ = cz(i)-W;
        CX2 = CX^2;   CY2 = CY^2;   CZ2 = CZ^2;
        C2  = CX2 + CY2 + CZ2;
        T(1,i)  = 1;
        T(2,i)  = CX;
        T(3,i)  = CY;
        T(4,i)  = CZ;
        T(5,i)  = C2-1;
        T(6,i)  = 3*CX2-C2;
        T(7,i)  = CY2-CZ2;
        T(8,i)  = CX*CY;
        T(9,i)  = CY*CZ;
        T(10,i) = CX*CZ;
        T(11,i) = CX*(3*C2-5);
        T(12,i) = CY*(3*C2-5);
        T(13,i) = CZ*(3*C2-5); 
        T(14,i) = CX*(CY2-CZ2);
        T(15,i) = CY*(CZ2-CX2);
        T(16,i) = CZ*(CX2-CY2);
        T(17,i) = 3*C2*C2 - 6*C2+1;
        T(18,i) = (2*C2-3)*(3*CX2-C2);
        T(19,i) = (2*C2-3)*(CY2-CZ2);
        
        CX  = cx(i);  CY  = cy(i);  CZ  = cz(i);
        CX2 = CX^2;   CY2 = CY^2;   CZ2 = CZ^2;
        C2  = CX2 + CY2 + CZ2;
    
        M(1,i)  = 1;
        M(2,i)  = CX;
        M(3,i)  = CY;
        M(4,i)  = CZ;
        M(5,i)  = C2-1;
        M(6,i)  = 3*CX2-C2;
        M(7,i)  = CY2-CZ2;
        M(8,i)  = CX*CY;
        M(9,i)  = CY*CZ;
        M(10,i) = CX*CZ;
        M(11,i) = CX*(3*C2-5);
        M(12,i) = CY*(3*C2-5);
        M(13,i) = CZ*(3*C2-5); 
        M(14,i) = CX*(CY2-CZ2);
        M(15,i) = CY*(CZ2-CX2);
        M(16,i) = CZ*(CX2-CY2);
        M(17,i) = 3*C2*C2 - 6*C2+1;
        M(18,i) = (2*C2-3)*(3*CX2-C2);
        M(19,i) = (2*C2-3)*(CY2-CZ2);
    else
        CX = cx(i)-U;
        CY = cy(i)-V;
        CZ = cz(i)-W;
        CX2 = CX^2;
        CY2 = CY^2;
        CZ2 = CZ^2;
        T(1,i) = 1;
        T(2,i) = CX;
        T(3,i) = CY;
        T(4,i) = CZ;
        T(5,i) = CX2+CY2+CZ2;
        T(6,i) = CX2-CY2;
        T(7,i) = CY2-CZ2;
        T(8,i) = CX*CY;
        T(9,i) = CX*CZ;
        T(10,i) = CY*CZ;
        T(11,i) = CX2*CY;
        T(12,i) = CX*CY2;
        T(13,i) = CX2*CZ;
        T(14,i) = CX*CZ2;
        T(15,i) = CY2*CZ;
        T(16,i) = CY*CZ2;
        T(17,i) = CX2*CY2;
        T(18,i) = CX2*CZ2;
        T(19,i) = CY2*CZ2;

        CX = cx(i);
        CY = cy(i);
        CZ = cz(i);
        CX2 = CX^2;
        CY2 = CY^2;
        CZ2 = CZ^2;
        M(1,i) = 1;
        M(2,i) = CX;
        M(3,i) = CY;
        M(4,i) = CZ;
        M(5,i) = CX2+CY2+CZ2;
        M(6,i) = CX2-CY2;
        M(7,i) = CY2-CZ2;
        M(8,i) = CX*CY;
        M(9,i) = CX*CZ;
        M(10,i) = CY*CZ;
        M(11,i) = CX2*CY;
        M(12,i) = CX*CY2;
        M(13,i) = CX2*CZ;
        M(14,i) = CX*CZ2;
        M(15,i) = CY2*CZ;
        M(16,i) = CY*CZ2;
        M(17,i) = CX2*CY2;
        M(18,i) = CX2*CZ2;
        M(19,i) = CY2*CZ2;
    end
end

if(central_moments)
    T = simplify(T);
    N = T*inv(M);
else
    T = M;
    N = eye(19,19);
end
%% Compute central moments
K_eq = simplify(T*feq);
Id = eye(19,19);
K_pre = sym(zeros(19,1));
syms k5_pre k6_pre k7_pre k8_pre k9_pre k24_pre k25_pre k26_pre real
K_pre(1) = R;
K_pre(6) = k5_pre;
K_pre(7) = k6_pre;
K_pre(8) = k7_pre;
K_pre(9) = k8_pre;
K_pre(10) = k9_pre;
%post-collision central moments
K_star = (Id-L)*K_pre + L*K_eq;
%post collision populations
syms k1 k2 k3 k4 k5 k6 k7 k8 k9 k10 k11 k12 k13 k14 k15 k16 k17 k18 real
K_sym = [R k1 k2 k3 k4 k5 k6 k7 k8 k9 k10 k11 k12 k13 k14 k15 k16 k17 k18];
for i=1:19
    if(K_star(i)~=sym(0))
        K_star(i) = K_sym(i);
    end
end
raw_moments = collect(simplify(N^(-1)*K_star), K_star)
syms r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 real
r = [r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18]'; %symbolic raw moments
f_post_collision_twosteps = simplify(M\r)