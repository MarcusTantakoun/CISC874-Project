(define (problem storage-0)
    (:domain Storage-Propositional)
    (:objects container0 - container crate0 crate1 crate2 crate3 - crate depot48 depot49 - depot hoist0 hoist1 hoist2 hoist3 hoist4 hoist5 - hoist container-0-0 container-0-1 container-0-2 container-0-3 depot48-1-1 depot48-1-2 depot48-2-1 depot48-2-2 depot49-1-1 depot49-1-2 depot49-1-3 depot49-2-1 depot49-2-2 - storearea loadarea - transitarea)
    (:init (at hoist0 depot48-1-1) (at hoist1 depot49-1-3) (at hoist2 depot49-2-2) (at hoist3 depot49-1-2) (at hoist4 depot49-1-1) (at hoist5 depot49-2-1) (available hoist0) (available hoist1) (available hoist2) (available hoist3) (available hoist4) (available hoist5) (clear depot48-1-2) (clear depot48-2-1) (clear depot48-2-2) (connected container-0-0 loadarea) (connected container-0-1 loadarea) (connected container-0-2 loadarea) (connected container-0-3 loadarea) (connected depot48-1-1 depot48-1-2) (connected depot48-1-1 depot48-2-1) (connected depot48-1-2 depot48-1-1) (connected depot48-1-2 depot48-2-2) (connected depot48-2-1 depot48-1-1) (connected depot48-2-1 depot48-2-2) (connected depot48-2-1 loadarea) (connected depot48-2-2 depot48-1-2) (connected depot48-2-2 depot48-2-1) (connected depot49-1-1 depot49-1-2) (connected depot49-1-1 depot49-2-1) (connected depot49-1-2 depot49-1-1) (connected depot49-1-2 depot49-1-3) (connected depot49-1-2 depot49-2-2) (connected depot49-1-3 depot49-1-2) (connected depot49-2-1 depot49-1-1) (connected depot49-2-1 depot49-2-2) (connected depot49-2-2 depot49-1-2) (connected depot49-2-2 depot49-2-1) (connected depot49-2-2 loadarea) (connected loadarea container-0-0) (connected loadarea container-0-1) (connected loadarea container-0-2) (connected loadarea container-0-3) (connected loadarea depot48-2-1) (connected loadarea depot49-2-2) (in container-0-0 container0) (in container-0-1 container0) (in container-0-2 container0) (in container-0-3 container0) (in crate0 container0) (in crate1 container0) (in crate2 container0) (in crate3 container0) (in depot48-1-1 depot48) (in depot48-1-2 depot48) (in depot48-2-1 depot48) (in depot48-2-2 depot48) (in depot49-1-1 depot49) (in depot49-1-2 depot49) (in depot49-1-3 depot49) (in depot49-2-1 depot49) (in depot49-2-2 depot49) (on crate0 container-0-0) (on crate1 container-0-1) (on crate2 container-0-2) (on crate3 container-0-3))
    (:goal (and (in crate0 depot48) (in crate1 depot49) (in crate2 depot49) (in crate3 depot49)))
)