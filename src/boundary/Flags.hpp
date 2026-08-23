#pragma once
//==============================================================================
//  Cell flags.
//
//  Boundary handling strategy: a boundary condition is an ALTERNATIVE COLLISION
//  OPERATOR applied to marked cells. Halfway bounce-back is "reflect everything
//  you received instead of colliding", which makes the wall sit exactly midway
//  between the last fluid node and the first solid node.
//
//  That gives one branch per NODE rather than one per direction, keeps the
//  population loads unconditional and coalesced, and generalises: Zou-He,
//  regularised, etc. all become other flag cases. Milestone 5 hoists these into
//  precomputed per-type index lists so the bulk kernel loses its branch entirely
//  -- the semantics below do not change when that happens.
//==============================================================================
#include "core/Types.hpp"

namespace lbm {

enum CellType : std::uint8_t {
  Fluid    = 0,   // collide
  Solid    = 1,   // halfway bounce-back
  Excluded = 2,   // not part of the simulation; skipped entirely
  RegWall  = 3,   // regularised velocity boundary, then collide
};

}  // namespace lbm
