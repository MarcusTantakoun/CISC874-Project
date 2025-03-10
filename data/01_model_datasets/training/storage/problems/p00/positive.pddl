(define (problem Storage-Propositional_0)
  (:domain Storage-Propositional)
  (:objects 
      depot48-1-1 depot48-1-2 depot48-1-3 depot48-1-4 depot48-1-5 depot48-1-6 depot48-2-1 depot48-2-2 depot48-2-3 depot48-2-4 depot48-2-5 depot48-2-6 depot48-3-1 depot48-3-2 depot48-3-3 depot48-3-4 depot48-3-5 depot48-3-6 depot48-4-1 depot48-4-2 depot48-4-3 depot48-4-4 depot48-4-5 depot48-4-6 depot48-5-1 depot48-5-2 depot48-5-3 depot48-5-4 depot48-5-5 depot48-5-6 depot48-6-1 depot48-6-2 depot48-6-3 depot48-6-4 depot48-6-5 depot48-6-6 - storearea
      hoist0 hoist1 hoist2 - hoist
      crate0 crate1 crate2 - crate
      container0 - container
      loadarea - transitarea
)
  (:init 
  (connected depot48-6-5 depot48-2-2)
  (connected depot48-6-2 depot48-4-1)
  (connected depot48-1-6 depot48-1-4)
  (connected depot48-6-1 depot48-5-5)
  (connected depot48-3-6 depot48-5-6)
  (connected depot48-1-5 depot48-2-1)
  (on crate2 container-0-0)
  (on crate0 container-1-0)
  (on crate1 container-2-0)
  (in depot48-1-1 depot48)
  (in depot48-2-1 depot48)
  (in depot48-3-1 depot48)
  (in depot48-4-1 depot48)
  (in depot48-5-1 depot48)
  (in depot48-6-1 depot48)
  (connected loadarea container-1-0)
  (connected loadarea container-2-0)
  (connected loadarea container-3-0)
  (at hoist0 depot48-6-2)
  (available hoist0)
  (at hoist1 depot48-3-1)
  (available hoist1)
  (at hoist2 depot48-5-5)
  (available hoist2)
  (clear depot48-1-1)
  (clear depot48-2-1)
  (clear depot48-3-1)
  (clear depot48-4-1)
  (clear depot48-5-1)
  (clear depot48-6-1)
)
  (:goal 
  (and
      (in crate0 depot48)
      (in crate1 depot48)
      (in crate2 depot48)
  ))
)