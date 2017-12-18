SHELL=/bin/bash
EMBEDDIR=/fs/project/PAS1315/projgroup7/embeddings

default:
	@echo "-- Make targets --"
	@echo "   preprocess_ddi_adapt SRC TRG PKL_DIR"
	@echo "   preprocess_ddi_adapt_xval SRC TRG PKL_DIR"

preprocess_ddi:
	@if [ -z "${SRC}" ] || [ -z "${PKL_DIR}" ]; then \
		echo "SRC and PKL_DIR must be specified"; \
		exit; \
	fi; \
	python preprocess.py \
		${PKL_DIR} \
		--source-embeddings ${EMBEDDIR}/${SRC} \
		--train files/train_ddi_reduced.txt \
		--validation files/validation_ddi.txt \
		--test files/test_ddi.txt \
		--dataset DDI

preprocess_ddi_valtest:
	@if [ -z "${SRC}" ] || [ -z "${PKL_DIR}" ]; then \
		echo "SRC and PKL_DIR must be specified"; \
		exit; \
	fi; \
	python preprocess.py \
		${PKL_DIR} \
		--source-embeddings ${EMBEDDIR}/${SRC} \
		--train files/train_ddi.txt \
		--validation files/test_ddi.txt \
		--dataset DDI

preprocess_ddi_adapt:
	@if [ -z "${SRC}" ] || [ -z "${TRG}" ] || [ -z "${PKL_DIR}" ]; then \
		echo "SRC, TRG, and PKL_DIR must be specified"; \
		exit; \
	fi; \
	python preprocess.py \
		${PKL_DIR} \
		--source-embeddings ${EMBEDDIR}/${SRC} \
		--target-embeddings ${EMBEDDIR}/${TRG} \
		--train files/train_ddi_reduced.txt \
		--validation files/validation_ddi.txt \
		--test files/test_ddi.txt \
		--dataset DDI

preprocess_ddi_adapt_valtest:
	@if [ -z "${SRC}" ] || [ -z "${TRG}" ] || [ -z "${PKL_DIR}" ]; then \
		echo "SRC, TRG, and PKL_DIR must be specified"; \
		exit; \
	fi; \
	python preprocess.py \
		${PKL_DIR} \
		--source-embeddings ${EMBEDDIR}/${SRC} \
		--target-embeddings ${EMBEDDIR}/${TRG} \
		--train files/train_ddi.txt \
		--validation files/test_ddi.txt \
		--dataset DDI
	

preprocess_ddi_adapt_xval:
	@if [ -z "${SRC}" ] || [ -z "${TRG}" ] || [ -z "${PKL_DIR}" ]; then \
		echo "SRC, TRG, and PKL_DIR must be specified"; \
		exit; \
	fi; \
	for fold in $$(seq 0 4); do \
		echo "Processing fold $${fold}"; \
		python preprocess.py \
			${PKL_DIR}/fold$${fold} \
			--source-embeddings ${EMBEDDIR}/${SRC} \
			--target-embeddings ${EMBEDDIR}/${TRG} \
			--train newData/reduced_train_train_ddi.txt_$${fold} \
			--validation newData/validation_train_ddi.txt_$${fold} \
			--dataset DDI; \
	done

preprocess_lstm_ddi_adapt:
	@if [ -z "${SRC}" ] || [ -z "${TRG}" ] || [ -z "${PKL_DIR}" ]; then \
		echo "SRC, TRG, and PKL_DIR must be specified"; \
		exit; \
	fi; \
	python preprocess_LSTM.py \
		${PKL_DIR} \
		--source-embeddings ${EMBEDDIR}/${SRC} \
		--target-embeddings ${EMBEDDIR}/${TRG} \
		--train files/train_ddi_new.txt \
		--validation files/test_ddi_new.txt \
		--dataset DDI_ALT

preprocess_lstm_ddi:
	@if [ -z "${SRC}" ] || [ -z "${PKL_DIR}" ]; then \
		echo "SRC and PKL_DIR must be specified"; \
		exit; \
	fi; \
	python preprocess_LSTM.py \
		${PKL_DIR} \
		--source-embeddings ${EMBEDDIR}/${SRC} \
		--train files/train_ddi_new.txt \
		--validation files/test_ddi_new.txt \
		--dataset DDI_ALT
