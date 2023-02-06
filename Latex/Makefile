DOCKERBIN := $(shell command -v docker 2>/dev/null)
ifdef DOCKERBIN # use docker if it is present
DOCKERIMG=danteev/texlive
DOCKER=docker run --rm -it -v $(shell pwd):/home $(DOCKERIMG)
endif

MAINFILE=thesis.tex

all: pdf

pdf:
	$(DOCKER) latexmk -shell-escape -pdf $(MAINFILE)

count:
	@$(DOCKER) texcount -relaxed -merge -char $(MAINFILE)

clean:
	$(DOCKER) latexmk -C

setup:
	docker pull $(DOCKERIMG):latest

