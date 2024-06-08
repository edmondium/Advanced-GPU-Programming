struct Mandelbrot {
	Mandelbrot(int h, int w): height{h}, width{w} {
		cudaMallocManaged(&data, height * width * sizeof(int));
	};
	~Mandelbrot(){
		cudaFree(data);
	}

	int height;
	int width;
	int* data;
};
