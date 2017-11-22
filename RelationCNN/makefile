SHELL=/bin/bash
EMBEDDIR=/fs/project/PAS1315/projgroup7/embeddings

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
