# These commands are here to help with local dev. They are not used in production.

build:
	docker build . -t basic-model-serve

#buildx:
#	docker buildx build --platform linux/amd64 . -t basic-model-serve

run:
	docker run --name basic-model-serve --rm -d -p 8000:8000 basic-model-serve

#only works if you only have one container running
kill:
	docker kill basic-model-serve