clear all
clc

central_moments = true;
ortho = true;   % production choice for D3Q27

syms U V W R omega k4_star k5_star k6_star k7_star k8_star...
     f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 f15 f16 f17 f18 f19 f20 f21 f22 f23 f24 f25 f26 real
f = [f0 f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13 f14 f15 f16 f17 f18 f19 f20 f21 f22 f23 f24 f25 f26];
% omega must relax the full set of independent second-order moments.  The two
% bases order those moments differently, so L has to follow the `ortho` toggle:
%   ortho : 5..7 = CX2-cs2, CY2-cs2, CZ2-cs2   8..10 = CXCY, CXCZ, CYCZ
%           -> omega on 5..10 (all six second-order moments)
%   non-ortho: 5..7 = CXCY, CXCZ, CYCZ   8..9 = CX2-CY2, CX2-CZ2   10 = C2
%           -> omega on 5..9 (the five deviatoric moments); position 10 is the
%              trace and must stay at rate 1, otherwise the bulk viscosity is
%              driven to zero as omega->2 and the scheme is unstable even at rest.
if(ortho)
    L = diag([1, 1, 1, 1, omega, omega, omega, omega, omega, omega, ones(1,17)]);
else
    L = diag([1, 1, 1, 1, omega, omega, omega, omega, omega, 1, ones(1,17)]);
end
cs = 1/sqrt(3);
cs2 = cs^2;
cs4 = cs^4;
cs6 = cs^6;
cs8 = cs^8;
cs10 = cs^10;
cs12 = cs^12;

%% ---------------------------------------------------------------------------
%  Direction ordering: ESOTERIC PULL convention.
%    index 0 is the rest velocity; opposite directions occupy ADJACENT indices,
%    so opp(i) = i+1 for odd i and i-1 for even i (1-based: 2<->3, 4<->5, ...).
%  Reordering directions is only a COLUMN permutation of T and M plus the same
%  permutation of f and feq, so every moment -- raw or central -- is unchanged;
%  the K_star / raw_moments output below is identical to the previous ordering
%  and only the per-direction f_post_collision lines come out permuted.
%  See esopull_ordering.m for the permutation vectors, and original/ for the
%  scripts as they were before this change.
%% ---------------------------------------------------------------------------
cx = [0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, -1, 1];
cy = [0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1];
cz = [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, 1, -1];
w = [8./27., 2./27., 2./27., 2./27., 2./27., 2./27., 2./27.,... 
	1./54., 1./54., 1./54., 1./54., 1./54., 1./54., 1./54., 1./54., 1./54., 1./54., 1./54., 1./54.,...
	1./216., 1./216., 1./216., 1./216., 1./216., 1./216., 1./216., 1./216.];
u2 = U*U+V*V+W*W;

T = sym(zeros(length(cx),length(cx)));
M = sym(zeros(length(cx),length(cx)));
feq = sym(zeros(length(cx),1));
%choose the maximum order of the Hermite polynomials N=2,3,4,5,6
for i=1:length(cx)
    first_order = U*cx(i)+V*cy(i)+W*cz(i);
    first_order = first_order/cs2;
    
    second_order = 4.5*(U*cx(i)+V*cy(i)+W*cz(i))^2-1.5*u2;
    
    third_order = (cx(i)^2-cs2)*cy(i)*U*U*V + (cx(i)^2-cs2)*cz(i)*U*U*W +...
                  (cy(i)^2-cs2)*cx(i)*U*V*V + (cz(i)^2-cs2)*cx(i)*U*W*W +...
                  (cz(i)^2-cs2)*cy(i)*V*W*W + (cy(i)^2-cs2)*cz(i)*V*V*W +...
              2*( cx(i)*cy(i)*cz(i)*U*V*W);  
    third_order = third_order/(2*cs6);

    fourth_order = (cx(i)^2-cs2)*(cy(i)^2-cs2)*U*U*V*V +...
                   (cx(i)^2-cs2)*(cz(i)^2-cs2)*U*U*W*W +...
                   (cy(i)^2-cs2)*(cz(i)^2-cs2)*V*V*W*W +...
                2*( cx(i)*cy(i)*(cz(i)^2-cs2)*U*V*W*W +...
                    cx(i)*(cy(i)^2-cs2)*cz(i)*U*V*V*W +...
                   (cx(i)^2-cs2)*cy(i)*cz(i)*U*U*V*W );
    fourth_order = fourth_order/(4*cs8);
    
    fifth_order = (cx(i)^2-cs2)*cy(i)*(cz(i)^2-cs2)*U*U*V*W*W +...
                  (cx(i)^2-cs2)*(cy(i)^2-cs2)*cz(i)*U*U*V*V*W +...
                  cx(i)*(cy(i)^2-cs2)*(cz(i)^2-cs2)*U*V*V*W*W;
    fifth_order = fifth_order/(4*cs10);

    sixth_order = (cx(i)^2-cs2)*(cy(i)^2-cs2)*(cz(i)^2-cs2)*U*U*V*V*W*W;
    sixth_order = sixth_order/(8*cs12);
    feq(i) = R*w(i)*(1+first_order + second_order + third_order +...
                         fourth_order + fifth_order + sixth_order);
   
    % Set the basis
    if(ortho)
        CX = cx(i)-U;
        CY = cy(i)-V;
        CZ = cz(i)-W;
        CX2 = CX*CX;
        CY2 = CY*CY;
        CZ2 = CZ*CZ;
        T(1,i) = 1;
        T(2,i) = CX;
        T(3,i) = CY;
        T(4,i) = CZ;
        T(5,i) = CX2-cs2;
        T(6,i) = CY2-cs2;
        T(7,i) = CZ2-cs2;
        T(8,i) = CX*CY;
        T(9,i) = CX*CZ;
        T(10,i) = CY*CZ;
        T(11,i) = CX2*CY-cs2*CY;
        T(12,i) = CX2*CZ-cs2*CZ;
        T(13,i) = CX*CY2-cs2*CX;
        T(14,i) = CX*CZ2-cs2*CX;
        T(15,i) = CY*CZ2-cs2*CY;
        T(16,i) = CY2*CZ-cs2*CZ;
        T(17,i) = CX*CY*CZ;
        T(18,i) = CX2*CY2 - cs2*(CX2+CY2) + cs4;
        T(19,i) = CX2*CZ2 - cs2*(CX2+CZ2) + cs4;
        T(20,i) = CY2*CZ2 - cs2*(CY2+CZ2) + cs4;
        T(21,i) = CX*CY*CZ2 - cs2*CX*CY;
        T(22,i) = CX*CY2*CZ - cs2*CX*CZ;
        T(23,i) = CX2*CY*CZ - cs2*CY*CZ;
        T(24,i) = CX2*CY*CZ2 - cs2*(CX2*CY+CY*CZ2) + cs4*CY;
        T(25,i) = CX2*CY2*CZ - cs2*(CX2*CZ+CY2*CZ) + cs4*CZ;
        T(26,i) = CX*CY2*CZ2 - cs2*(CX*CY2+CX*CZ2) + cs4*CX;
        T(27,i) = CX2*CY2*CZ2 - cs2*(CX2*CY2+CX2*CZ2+CY2*CZ2) + cs4*(CX2+CY2+CZ2) - cs6;

        CX = cx(i);
        CY = cy(i);
        CZ = cz(i);
        CX2 = CX*CX;
        CY2 = CY*CY;
        CZ2 = CZ*CZ;
        M(1,i) = 1;
        M(2,i) = CX;
        M(3,i) = CY;
        M(4,i) = CZ;
        M(5,i) = CX2-cs2;
        M(6,i) = CY2-cs2;
        M(7,i) = CZ2-cs2;
        M(8,i) = CX*CY;
        M(9,i) = CX*CZ;
        M(10,i) = CY*CZ;
        M(11,i) = CX2*CY-cs2*CY;
        M(12,i) = CX2*CZ-cs2*CZ;
        M(13,i) = CX*CY2-cs2*CX;
        M(14,i) = CX*CZ2-cs2*CX;
        M(15,i) = CY*CZ2-cs2*CY;
        M(16,i) = CY2*CZ-cs2*CZ;
        M(17,i) = CX*CY*CZ;
        M(18,i) = CX2*CY2 - cs2*(CX2+CY2) + cs4;
        M(19,i) = CX2*CZ2 - cs2*(CX2+CZ2) + cs4;
        M(20,i) = CY2*CZ2 - cs2*(CY2+CZ2) + cs4;
        M(21,i) = CX*CY*CZ2 - cs2*CX*CY;
        M(22,i) = CX*CY2*CZ - cs2*CX*CZ;
        M(23,i) = CX2*CY*CZ - cs2*CY*CZ;
        M(24,i) = CX2*CY*CZ2 - cs2*(CX2*CY+CY*CZ2) + cs4*CY;
        M(25,i) = CX2*CY2*CZ - cs2*(CX2*CZ+CY2*CZ) + cs4*CZ;
        M(26,i) = CX*CY2*CZ2 - cs2*(CX*CY2+CX*CZ2) + cs4*CX;
        M(27,i) = CX2*CY2*CZ2 - cs2*(CX2*CY2+CX2*CZ2+CY2*CZ2) + cs4*(CX2+CY2+CZ2) - cs6;
    else
        CX = cx(i)-U;
        CY = cy(i)-V;
        CZ = cz(i)-W;
        T(1,i) = 1;
        T(2,i) = CX;
        T(3,i) = CY;
        T(4,i) = CZ;
        T(5,i) = CX*CY;
        T(6,i) = CX*CZ;
        T(7,i) = CY*CZ;
        T(8,i) = CX*CX-CY*CY;
        T(9,i) = CX*CX-CZ*CZ;
        T(10,i) = CX*CX+CY*CY+CZ*CZ;
        T(11,i) = CX*CY*CY+CX*CZ*CZ;
        T(12,i) = CX*CX*CY+CY*CZ*CZ;
        T(13,i) = CX*CX*CZ+CY*CY*CZ;
        T(14,i) = CX*CY*CY-CX*CZ*CZ;
        T(15,i) = CX*CX*CY-CY*CZ*CZ;
        T(16,i) = CX*CX*CZ-CY*CY*CZ;
        T(17,i) = CX*CY*CZ;
        T(18,i) = CX*CX*CY*CY+CX*CX*CZ*CZ+CY*CY*CZ*CZ;
        T(19,i) = CX*CX*CY*CY+CX*CX*CZ*CZ-CY*CY*CZ*CZ;
        T(20,i) = CX*CX*CY*CY-CX*CX*CZ*CZ;
        T(21,i) = CX*CX*CY*CZ;
        T(22,i) = CX*CY*CY*CZ;
        T(23,i) = CX*CY*CZ*CZ;
        T(24,i) = CX*CY*CY*CZ*CZ;
        T(25,i) = CX*CX*CY*CZ*CZ;
        T(26,i) = CX*CX*CY*CY*CZ;
        T(27,i) = CX*CX*CY*CY*CZ*CZ;

        CX = cx(i);
        CY = cy(i);
        CZ = cz(i);
        M(1,i) = 1;
        M(2,i) = CX;
        M(3,i) = CY;
        M(4,i) = CZ;
        M(5,i) = CX*CY;
        M(6,i) = CX*CZ;
        M(7,i) = CY*CZ;
        M(8,i) = CX*CX-CY*CY;
        M(9,i) = CX*CX-CZ*CZ;
        M(10,i) = CX*CX+CY*CY+CZ*CZ;
        M(11,i) = CX*CY*CY+CX*CZ*CZ;
        M(12,i) = CX*CX*CY+CY*CZ*CZ;
        M(13,i) = CX*CX*CZ+CY*CY*CZ;
        M(14,i) = CX*CY*CY-CX*CZ*CZ;
        M(15,i) = CX*CX*CY-CY*CZ*CZ;
        M(16,i) = CX*CX*CZ-CY*CY*CZ;
        M(17,i) = CX*CY*CZ;
        M(18,i) = CX*CX*CY*CY+CX*CX*CZ*CZ+CY*CY*CZ*CZ;
        M(19,i) = CX*CX*CY*CY+CX*CX*CZ*CZ-CY*CY*CZ*CZ;
        M(20,i) = CX*CX*CY*CY-CX*CX*CZ*CZ;
        M(21,i) = CX*CX*CY*CZ;
        M(22,i) = CX*CY*CY*CZ;
        M(23,i) = CX*CY*CZ*CZ;
        M(24,i) = CX*CY*CY*CZ*CZ;
        M(25,i) = CX*CX*CY*CZ*CZ;
        M(26,i) = CX*CX*CY*CY*CZ;
        M(27,i) = CX*CX*CY*CY*CZ*CZ;
    end
end
% Compute central moments
if(central_moments)
    T = simplify(T);
    N = simplify(T/M); %shift matrix
else
    T = M;
    N = eye(27,27);
end

K_eq = simplify(T*feq);
Id = eye(length(cx),length(cx));
K_pre = sym(zeros(length(cx),1));
syms k4_pre k5_pre k6_pre k7_pre k8_pre k9_pre real
K_pre(1) = R;
% Every position that L relaxes at omega must carry its pre-collision value,
% otherwise (Id-L)*K_pre drops the (1-omega)*k_pre term there and that moment is
% driven to equilibrium at rate 1 instead of omega -- which makes the viscosity
% direction-dependent. The two bases order the second-order moments differently,
% so the assignment has to follow the `ortho` toggle exactly as L does.
if(ortho)
    % 5..10 = CX2-cs2, CY2-cs2, CZ2-cs2, CXCY, CXCZ, CYCZ  (all six at omega)
    K_pre(5)  = k4_pre;
    K_pre(6)  = k5_pre;
    K_pre(7)  = k6_pre;
    K_pre(8)  = k7_pre;
    K_pre(9)  = k8_pre;
    K_pre(10) = k9_pre;
else
    % 5..9 = CXCY, CXCZ, CYCZ, CX2-CY2, CX2-CZ2  (omega); 10 = C2 stays at 1
    K_pre(5) = k4_pre;
    K_pre(6) = k5_pre;
    K_pre(7) = k6_pre;
    K_pre(8) = k7_pre;
    K_pre(9) = k8_pre;
end
%post-collision central moments
K_star = (Id-L)*K_pre + L*K_eq

%post collision populations
syms k1 k2 k3 k4 k5 k6 k7 k8 k9 k10 k11 k12 k13 k14 k15 k16 k17 k18 k19...
     k20 k21 k22 k23 k24 k25 k26 real
K_sym = [R k1 k2 k3 k4 k5 k6 k7 k8 k9 k10 k11 k12 k13 k14 k15 k16 k17 k18 k19...
         k20 k21 k22 k23 k24 k25 k26];
for i=1:length(cx)
    if(K_star(i)~=sym(0))
        K_star(i) = K_sym(i);
    end
end
raw_moments = collect(simplify(N \ K_star), K_star)
syms r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19...
     r20 r21 r22 r23 r24 r25 r26 real
r = [r0 r1 r2 r3 r4 r5 r6 r7 r8 r9 r10 r11 r12 r13 r14 r15 r16 r17 r18 r19...
     r20 r21 r22 r23 r24 r25 r26]'; %symbolic raw moments
f_post_collision_twosteps = collect(simplify(M\r),K_star);
f_post_collision_twosteps = collect(f_post_collision_twosteps, 1/8);
f_post_collision_twosteps = collect(f_post_collision_twosteps, -1/8);
f_post_collision_twosteps = collect(f_post_collision_twosteps, 1/4);
f_post_collision_twosteps = collect(f_post_collision_twosteps, -1/4);
f_post_collision_twosteps = collect(f_post_collision_twosteps, 1/2);
f_post_collision_twosteps = collect(f_post_collision_twosteps, -1/2);
f_post_collision_twosteps = collect(f_post_collision_twosteps, 1/6);
f_post_collision_twosteps = collect(f_post_collision_twosteps, -1/6);
f_post_collision_twosteps = collect(f_post_collision_twosteps, 1/3);
f_post_collision_twosteps = collect(f_post_collision_twosteps, -1/3);
f_post_collision_twosteps = collect(f_post_collision_twosteps, 1/18);
f_post_collision_twosteps = collect(f_post_collision_twosteps, -1/18);
f_post_collision_twosteps = collect(f_post_collision_twosteps, 1/24);
f_post_collision_twosteps = collect(f_post_collision_twosteps, -1/24);
f_post_collision_twosteps = collect(f_post_collision_twosteps, 1/12);
f_post_collision_twosteps = collect(f_post_collision_twosteps, -1/12);
f_post_collision_twosteps = collect(f_post_collision_twosteps, 1/72);
f_post_collision_twosteps = collect(f_post_collision_twosteps, -1/72);
f_post_collision_twosteps = collect(f_post_collision_twosteps, 2/3);
f_post_collision_twosteps = collect(f_post_collision_twosteps, -2/3);
f_post_collision_twosteps = collect(f_post_collision_twosteps, -4/9);
f_post_collision_twosteps = collect(f_post_collision_twosteps, 1/9);
f_post_collision_twosteps = collect(f_post_collision_twosteps, -1/9)
