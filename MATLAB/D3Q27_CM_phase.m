clear all
clc
%% ===========================================================================
%  D3Q27 CENTRAL-MOMENT COLLISION FOR THE CONSERVATIVE ALLEN-CAHN PHASE FIELD
%
%  The symbolic companion to src/collision/PhaseFieldCentralMoments.hpp, which
%  implements De Rosis & Enan, Phys. Fluids 33, 043315 (2021), Sec. II.D on
%  D3Q27 rather than their D3Q19.
%
%  WHAT THIS IS FOR. The phase field's collision is short enough to write out
%  by hand and short enough to get wrong by hand, and the two ways of getting
%  it wrong are both silent -- neither crashes, and both give a converged
%  interface of the wrong width. This script derives the whole thing from the
%  basis, so the two traps below are demonstrated rather than asserted:
%
%    1. THE EQUILIBRIUM COLLAPSES. Their Eq. (54) lists five nonzero
%       equilibrium central moments in the MONOMIAL basis. In the shifted
%       product basis used here, phi_2 = C^2 - cs2, all but the zeroth are
%       identically zero, so k_eq = (phi, 0, ..., 0) and no equilibrium table
%       is needed at all. Section 4 below shows both and the map between them.
%
%    2. THE SOURCE IS THREE TERMS HERE AND NINE IN THEIR PAPER, AND ONLY AT
%       u = 0. Their Eq. (61) has R_{1,2,3} = F and six third-order
%       R_{aab} = F_b cs2. In the shifted basis each of those six contributes
%       cs4 A_b - cs2*cs2 A_b = 0, so three survive. Adding them here would
%       double-count; dropping them from a monomial code would lose them.
%
%       BUT THAT CANCELLATION HOLDS AT u = 0 ONLY. Section 5 computes the
%       source moments at general u as well, and there 26 of the 27 are
%       nonzero in BOTH bases. Their drivers add a u-INDEPENDENT source, and so
%       does PhaseFieldCentralMoments.hpp; both therefore truncate the source's
%       central moments at u = 0. That is a real approximation shared by the
%       two implementations, not an identity, and Section 5 prints both so the
%       difference between "cancels" and "is truncated" stays visible.
%
%  Set `emit_code` to print the post-collision populations as C++.
%% ===========================================================================
emit_code = false;     % true -> print collide() as C++ expressions
check_only = true;     % true -> run the identity checks and report

syms U V W PHI omega real
syms Ax Ay Az real                       % the anti-diffusion vector, theta*n
syms g0 g1 g2 g3 g4 g5 g6 g7 g8 g9 g10 g11 g12 g13 g14 ...
     g15 g16 g17 g18 g19 g20 g21 g22 g23 g24 g25 g26 real
g = [g0 g1 g2 g3 g4 g5 g6 g7 g8 g9 g10 g11 g12 g13 g14 ...
     g15 g16 g17 g18 g19 g20 g21 g22 g23 g24 g25 g26].';

cs2 = sym(1)/3;  cs4 = cs2^2;

%% ---------------------------------------------------------------------------
%  Direction ordering: ESOTERIC PULL, as in D3Q27_CM.m and Lattices.hpp.
%  Opposite directions occupy adjacent indices, opp(i) = i+1 for odd i. That
%  is a contract the streaming depends on; reordering here is only a column
%  permutation of T and leaves every moment unchanged.
%% ---------------------------------------------------------------------------
cx = [0, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1,-1, 1];
cy = [0, 0, 0, 1,-1, 0, 0, 1,-1,-1, 1, 0, 0, 0, 0, 1,-1, 1,-1, 1,-1, 1,-1,-1, 1, 1,-1];
cz = [0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 1,-1,-1, 1, 1,-1,-1, 1, 1,-1,-1, 1, 1,-1, 1,-1];
w  = [8/27, 2/27,2/27,2/27,2/27,2/27,2/27, ...
      1/54,1/54,1/54,1/54,1/54,1/54,1/54,1/54,1/54,1/54,1/54,1/54, ...
      1/216,1/216,1/216,1/216,1/216,1/216,1/216,1/216];
Q = 27;

%% ---------------------------------------------------------------------------
%  1. THE TWO BASES.
%
%  SHIFTED (what ProductBasis.hpp uses, and what the C++ collides in):
%       basis(a,b,c) = phi_a(CX) phi_b(CY) phi_c(CZ),
%       phi_0 = 1,  phi_1 = C,  phi_2 = C^2 - cs2.
%  MONOMIAL (what their paper tabulates):
%       basis(a,b,c) = CX^a CY^b CZ^c.
%
%  Both are tensor products over a,b,c in {0,1,2}, so the row index is
%  n = (a*3+b)*3+c, matching Basis::index_of in the C++ exactly. The two differ
%  ONLY by the -cs2 in phi_2, and that single term is what moves the physics
%  between slots.
%% ---------------------------------------------------------------------------
idx = @(a,b,c) (a*3+b)*3+c+1;            % 1-based, same layout as index_of

Tsh  = sym(zeros(Q,Q));
Tmon = sym(zeros(Q,Q));
lbl  = strings(Q,1);
for i = 1:Q
    CX = cx(i)-U;  CY = cy(i)-V;  CZ = cz(i)-W;
    ph  = {sym(1), CX, CX^2-cs2};        % shifted, per direction
    ph2 = {sym(1), CY, CY^2-cs2};
    ph3 = {sym(1), CZ, CZ^2-cs2};
    mo  = {sym(1), CX, CX^2};            % monomial
    mo2 = {sym(1), CY, CY^2};
    mo3 = {sym(1), CZ, CZ^2};
    for a = 0:2
        for b = 0:2
            for c = 0:2
                n = idx(a,b,c);
                Tsh(n,i)  = ph{a+1}*ph2{b+1}*ph3{c+1};
                Tmon(n,i) = mo{a+1}*mo2{b+1}*mo3{c+1};
                if i == 1
                    lbl(n) = sprintf('(%d,%d,%d)', a, b, c);
                end
            end
        end
    end
end

%% ---------------------------------------------------------------------------
%  2. THE EQUILIBRIUM, as a product form rather than a Hermite series.
%
%  Per direction the one-dimensional weights are fixed by three conditions --
%  sum = 1, first moment = u, second moment = u^2 + cs2 -- and the D3Q27
%  equilibrium is their product times phi. Writing it this way is what makes
%  the central moments diagonal; it is the same statement the fluid operator's
%  banner makes about the product-form Maxwellian.
%% ---------------------------------------------------------------------------
w1 = @(c,u) (c==0)*(sym(2)/3 - u^2) + (c==1)*((u^2+cs2+u)/2) + (c==-1)*((u^2+cs2-u)/2);
geq = sym(zeros(Q,1));
for i = 1:Q
    geq(i) = PHI * w1(cx(i),U) * w1(cy(i),V) * w1(cz(i),W);
end

%% ---------------------------------------------------------------------------
%  3. THE SOURCE, built exactly as PhaseFieldBGK builds it so that the two
%  operators share it and any difference between them is the collision alone.
%       S_i = (1 - omega/2) w_i (c_i . A),      A = theta n = (4/W) phi(1-phi) n
%% ---------------------------------------------------------------------------
S = sym(zeros(Q,1));
for i = 1:Q
    S(i) = (1 - omega/2) * w(i) * (cx(i)*Ax + cy(i)*Ay + cz(i)*Az);
end

%% ---------------------------------------------------------------------------
%  4. TRAP ONE: where the equilibrium lives in each basis.
%% ---------------------------------------------------------------------------
keq_sh  = simplify(expand(Tsh  * geq));
keq_mon = simplify(expand(Tmon * geq));

fprintf('\n== equilibrium central moments ==\n');
fprintf('SHIFTED basis (what the C++ relaxes toward):\n');
nz = find(keq_sh ~= 0);
for n = nz.'
    fprintf('   k_eq%-8s = %s\n', lbl(n), char(keq_sh(n)));
end
fprintf('   -- %d nonzero of %d. Everything above the zeroth is ZERO,\n', numel(nz), Q);
fprintf('      which is why collide() needs no equilibrium table.\n');

fprintf('MONOMIAL basis (what their Eq. (54) tabulates):\n');
nzm = find(keq_mon ~= 0);
for n = nzm.'
    fprintf('   k_eq%-8s = %s\n', lbl(n), char(keq_mon(n)));
end
fprintf('   -- %d nonzero. Their five, plus the D3Q27 corners D3Q19 lacks.\n', numel(nzm));

%% ---------------------------------------------------------------------------
%  5. TRAP TWO: where the source lives in each basis.
%% ---------------------------------------------------------------------------
R_sh  = simplify(expand(Tsh  * S));
R_mon = simplify(expand(Tmon * S));

fprintf('\n== source central moments ==\n');
fprintf('SHIFTED: ');
nz = find(R_sh ~= 0);
fprintf('%d nonzero -- %s\n', numel(nz), strjoin(cellstr(lbl(nz)).', ', '));
for n = nz.'
    fprintf('   R%-8s = %s\n', lbl(n), char(R_sh(n)));
end
fprintf('MONOMIAL: %d nonzero.\n', numel(find(R_mon ~= 0)));

% ... and the same thing at u = 0, which is what both implementations use.
z = [U V W];  zv = [0 0 0];
R_sh0  = simplify(subs(R_sh,  z, zv));
R_mon0 = simplify(subs(R_mon, z, zv));
fprintf('\nAT u = 0, which is the truncation BOTH codes make:\n');
fprintf('   SHIFTED  %d nonzero -- the three first-order slots and nothing else\n', ...
        numel(find(R_sh0 ~= 0)));
fprintf('   MONOMIAL %d nonzero -- their nine on D3Q19, plus the corners D3Q27 adds\n', ...
        numel(find(R_mon0 ~= 0)));
fprintf('   monomial R(2,1,0)|u=0 = %s   <- their cs4 A_y\n', char(R_mon0(idx(2,1,0))));
fprintf('   shifted  R(2,1,0)|u=0 = %s   <- cs4 Ay - cs2*cs2 Ay, cancelled\n', ...
        char(R_sh0(idx(2,1,0))));
fprintf('   AT GENERAL u NEITHER IS SPARSE: %d and %d nonzero. The collision in\n', ...
        numel(find(R_sh ~= 0)), numel(find(R_mon ~= 0)));
fprintf('   Section 6 uses the u = 0 form, as their drivers do.\n');

%% ---------------------------------------------------------------------------
%  6. THE COLLISION.  K = diag(1, omega, omega, omega, 1, ..., 1), their
%  Eq. (55): the zeroth is conserved, the three first-order moments relax at
%  omega_phi and set the mobility M = cs2 (1/omega - 1/2), and every higher
%  moment goes straight to equilibrium -- which here means to zero.
%% ---------------------------------------------------------------------------
k = Tsh * g;                                  % pre-collision central moments
kstar = sym(zeros(Q,1));
kstar(idx(0,0,0)) = PHI;                                  % conserved
R0 = simplify(subs(R_sh, [U V W], [0 0 0]));      % the u = 0 source, as they use
kstar(idx(1,0,0)) = (1-omega)*k(idx(1,0,0)) + R0(idx(1,0,0));
kstar(idx(0,1,0)) = (1-omega)*k(idx(0,1,0)) + R0(idx(0,1,0));
kstar(idx(0,0,1)) = (1-omega)*k(idx(0,0,1)) + R0(idx(0,0,1));
% every other slot stays zero: equilibrium plus a source that vanishes there

gstar = simplify(expand(Tsh \ kstar));        % back to populations

%% ---------------------------------------------------------------------------
%  7. CHECKS. Each of these is a statement the C++ makes; if one fails, the
%  C++ and this script disagree and one of them is wrong.
%% ---------------------------------------------------------------------------
if check_only
    fprintf('\n== checks ==\n');
    ok = @(c,s) fprintf('   %-4s %s\n', string(c)=="1"|c==true, s);

    c1 = isAlways(simplify(sum(gstar) - PHI) == 0, 'Unknown','false');
    ok(c1, 'phi is conserved by the collision');

    m1 = simplify(expand(sum(gstar.*cx.') - ( (1-omega)*sum(g.*cx.') ...
         + omega*PHI*U + (1-omega/2)*cs2*Ax + omega*0 )));
    % the first RAW moment: m1* = (1-omega) m1 + omega phi U + (1-omega/2) cs2 Ax
    c2 = isAlways(m1 == 0, 'Unknown','false');
    ok(c2, 'first raw moment is (1-w)m1 + w phi U + (1-w/2) cs2 Ax');

    c3 = isAlways(simplify(keq_sh(idx(0,0,0)) - PHI) == 0,'Unknown','false');
    ok(c3, 'zeroth equilibrium central moment is phi');

    c4 = all(arrayfun(@(n) isAlways(keq_sh(n)==0,'Unknown','false'), ...
                      setdiff(1:Q, idx(0,0,0))));
    ok(c4, 'ALL higher equilibrium central moments vanish (shifted basis)');

    c5 = isAlways(simplify(R_mon(idx(2,1,0)) - (1-omega/2)*cs4*Ay)==0,'Unknown','false');
    ok(c5, 'monomial R(2,1,0) = (1-w/2) cs4 Ay, their Eq. (61)');

    c6 = all(arrayfun(@(n) isAlways(R_sh0(n)==0,'Unknown','false'), ...
             setdiff(1:Q, [idx(1,0,0) idx(0,1,0) idx(0,0,1)])));
    ok(c6, 'AT u=0 the shifted source is confined to the first-order slots');
    c6b = ~all(arrayfun(@(n) isAlways(R_sh(n)==0,'Unknown','false'), ...
             setdiff(1:Q, [idx(1,0,0) idx(0,1,0) idx(0,0,1)])));
    ok(c6b, 'and at general u it is NOT -- both codes truncate here');

    % the mobility, both ways round
    Msym = cs2*(1/omega - sym(1)/2);
    c7 = isAlways(simplify(1/(Msym/cs2 + sym(1)/2) - omega)==0,'Unknown','false');
    ok(c7, 'M = cs2(1/w - 1/2) inverts to w = 1/(M/cs2 + 1/2)');
end

%% ---------------------------------------------------------------------------
%  8. C++ emission, if wanted.
%% ---------------------------------------------------------------------------
if emit_code
    fprintf('\n== post-collision populations ==\n');
    for i = 1:Q
        fprintf('h[%2d] = %s;\n', i-1, char(gstar(i)));
    end
end
