% ============================================================================
% Esoteric-Pull-conforming direction orderings.
%
% Requirement: rest velocity at index 0, then opposite directions in ADJACENT
% pairs, i.e. opp(1)=2, opp(3)=4, opp(5)=6, ...  This lets the streaming kernel
% use  opp(i) = i+1 (i odd) / i-1 (i even)  with no lookup table.
%
% Reordering directions is a COLUMN permutation of T and M and a permutation of
% the entries of f and feq.  Every moment (raw or central) is invariant, so the
% symbolic K_star / raw_moments output is UNCHANGED -- only the per-direction
% f_post_collision lines come out permuted.  Just paste these arrays over the
% cx/cy/cz/w definitions at the top of each script and re-run.
%
% perm_new2old(k) gives the OLD (current-script, 1-based) index now sitting at
% new slot k, so:  f_new = f_old(perm_new2old)  converts existing output.
% ============================================================================

%% ---------------- D2Q5 ----------------
cx = [  0,  1, -1,  0,  0 ];
cy = [  0,  0,  0,  1, -1 ];
w  = [ 1/3, 1/6, 1/6, 1/6, 1/6 ];
cs2 = 1/3;
% (not present in the current scripts -- new lattice)

%% ---------------- D2Q9 ----------------
cx = [  0,  1, -1,  0,  0,  1, -1,  1, -1 ];
cy = [  0,  0,  0,  1, -1,  1, -1, -1,  1 ];
w  = [ 4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36 ];
cs2 = 1/3;
perm_new2old = [ 1, 2, 4, 3, 5, 6, 8, 9, 7 ];  % 1-based
perm_old2new = [ 1, 2, 4, 3, 5, 6, 9, 7, 8 ];  % 1-based

%% ---------------- D3Q7 ----------------
cx = [  0,  1, -1,  0,  0,  0,  0 ];
cy = [  0,  0,  0,  1, -1,  0,  0 ];
cz = [  0,  0,  0,  0,  0,  1, -1 ];
w  = [ 1/4, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8 ];
cs2 = 1/4;
% (not present in the current scripts -- new lattice)

%% ---------------- D3Q19 ----------------
cx = [  0,  1, -1,  0,  0,  0,  0,  1, -1,  1, -1,  1, -1,  1, -1,  0,  0,  0,  0 ];
cy = [  0,  0,  0,  1, -1,  0,  0,  1, -1, -1,  1,  0,  0,  0,  0,  1, -1,  1, -1 ];
cz = [  0,  0,  0,  0,  0,  1, -1,  0,  0,  0,  0,  1, -1, -1,  1,  1, -1, -1,  1 ];
w  = [ 1/3, 1/18, 1/18, 1/18, 1/18, 1/18, 1/18, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36 ];
cs2 = 1/3;
perm_new2old = [ 1, 2, 3, 4, 5, 6, 7, 8, 11, 10, 9, 12, 15, 14, 13, 16, 19, 18, 17 ];  % 1-based
perm_old2new = [ 1, 2, 3, 4, 5, 6, 7, 8, 11, 10, 9, 12, 15, 14, 13, 16, 19, 18, 17 ];  % 1-based

%% ---------------- D3Q27 ----------------
cx = [  0,  1, -1,  0,  0,  0,  0,  1, -1,  1, -1,  1, -1,  1, -1,  0,  0,  0,  0,  1, -1,  1, -1,  1, -1, -1,  1 ];
cy = [  0,  0,  0,  1, -1,  0,  0,  1, -1, -1,  1,  0,  0,  0,  0,  1, -1,  1, -1,  1, -1,  1, -1, -1,  1,  1, -1 ];
cz = [  0,  0,  0,  0,  0,  1, -1,  0,  0,  0,  0,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1, -1,  1,  1, -1,  1, -1 ];
w  = [ 8/27, 2/27, 2/27, 2/27, 2/27, 2/27, 2/27, 1/54, 1/54, 1/54, 1/54, 1/54, 1/54, 1/54, 1/54, 1/54, 1/54, 1/54, 1/54, 1/216, 1/216, 1/216, 1/216, 1/216, 1/216, 1/216, 1/216 ];
cs2 = 1/3;
perm_new2old = [ 1, 2, 3, 4, 5, 6, 7, 8, 11, 10, 9, 12, 15, 14, 13, 16, 19, 18, 17, 20, 27, 24, 23, 22, 25, 21, 26 ];  % 1-based
perm_old2new = [ 1, 2, 3, 4, 5, 6, 7, 8, 11, 10, 9, 12, 15, 14, 13, 16, 19, 18, 17, 20, 26, 24, 23, 22, 25, 27, 21 ];  % 1-based
