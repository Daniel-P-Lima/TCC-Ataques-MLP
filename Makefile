.PHONY: help install-deps run-svm clean-svm

help:
	@echo "make install-deps   # install minimal deps into .venv"
	@echo "make run-svm        # run the SVM script using .venv python"
	@echo "make clean-svm      # remove generated SVM outputs"

install-deps:
	@.venv/bin/python -m pip install --upgrade pip
	@.venv/bin/python -m pip install numpy pandas scikit-learn matplotlib joblib

install: install-deps

run-svm:
	@echo "Running SVM..."
	@.venv/bin/python SVM/svm.py

clean-svm:
	@rm -f SVM/svm_iot23.joblib SVM/svm_scaler.joblib SVM/svm_label_encoder.joblib SVM/svm_roc.png
	@echo "SVM outputs removed"
