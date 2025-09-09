
(cl:in-package :asdf)

(defsystem "precise_pushing_bridge-srv"
  :depends-on (:roslisp-msg-protocol :roslisp-utils )
  :components ((:file "_package")
    (:file "Step" :depends-on ("_package_Step"))
    (:file "_package_Step" :depends-on ("_package"))
  ))