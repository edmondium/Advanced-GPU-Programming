ifndef VERBOSE
.SILENT:
endif

all: t00 t01 t02 t03 t04 t05 t06 t07 t08 h01 h02

SUBMIT_CMD = srun --partition=dc-gpu --account training 2418 --reservation training2418

t00: t00.cpp
	cmake --preset system -B build -S . --log-level=WARNING
	cmake --build build --target $@ -- -s
	$(SUBMIT_CMD) ./build/$@

t00.nsys: t00.cpp
	cmake --preset system -B build -S . --log-level=WARNING
	cmake --build build --target $@ -- -s
	$(SUBMIT_CMD) nsys profile --force-overwrite true --output $@ ./build/$@

t01: t01.cpp
	cmake --preset system -B build -S . --log-level=WARNING
	cmake --build build --target $@ -- -s
	$(SUBMIT_CMD) ./build/$@

t01.nsys: t01.cpp
	cmake --preset system -B build -S . --log-level=WARNING
	cmake --build build --target $@ -- -s
	$(SUBMIT_CMD) nsys profile --force-overwrite true --output $@ ./build/$@


t02: t02.cpp
	cmake --preset system -B build -S . --log-level=WARNING
	cmake --build build --target $@ -- -s
	$(SUBMIT_CMD) ./build/$@

t02.nsys: t02.cpp
	cmake --preset system -B build -S . --log-level=WARNING
	cmake --build build --target $@ -- -s
	$(SUBMIT_CMD) nsys profile --force-overwrite true --output $@ ./build/$@

t03: t03.cpp
	cmake --preset system -B build -S . --log-level=WARNING
	cmake --build build --target $@ -- -s
	$(SUBMIT_CMD) ./build/$@

t03.nsys: t03.cpp
	cmake --preset system -B build -S . --log-level=WARNING
	cmake --build build --target $@ -- -s
	$(SUBMIT_CMD) nsys profile --force-overwrite true --output $@ ./build/$@


t04: t04.cpp
	cmake --preset system -B build -S . --log-level=WARNING
	cmake --build build --target $@ -- -s
	$(SUBMIT_CMD) ./build/$@

t04.nsys: t04.cpp
	cmake --preset system -B build -S . --log-level=WARNING
	cmake --build build --target $@ -- -s
	$(SUBMIT_CMD) nsys profile --force-overwrite true --output $@ ./build/$@

t05: t05.cpp
	cmake --preset system -B build -S . --log-level=WARNING
	cmake --build build --target $@ -- -s
	$(SUBMIT_CMD) ./build/$@

t05.nsys: t05.cpp
	cmake --preset system -B build -S . --log-level=WARNING
	cmake --build build --target $@ -- -s
	$(SUBMIT_CMD) nsys profile --force-overwrite true --output $@ ./build/$@


t06: t06.cpp
	cmake --preset system -B build -S . --log-level=WARNING
	cmake --build build --target $@ -- -s
	$(SUBMIT_CMD) ./build/$@

t06.nsys: t06.cpp
	cmake --preset system -B build -S . --log-level=WARNING
	cmake --build build --target $@ -- -s
	$(SUBMIT_CMD) nsys profile --force-overwrite true --output $@ ./build/$@


t07: t07.cpp
	cmake --preset system -B build -S . --log-level=WARNING
	cmake --build build --target $@ -- -s
	$(SUBMIT_CMD) ./build/$@

t07.nsys: t07.cpp
	cmake --preset system -B build -S . --log-level=WARNING
	cmake --build build --target $@ -- -s
	$(SUBMIT_CMD) nsys profile --force-overwrite true --output $@ ./build/$@

t08: t08.cpp
	cmake --preset system -B build -S . --log-level=WARNING
	cmake --build build --target $@ -- -s
	$(SUBMIT_CMD) ./build/$@

t08.nsys: t08.cpp
	cmake --preset system -B build -S . --log-level=WARNING
	cmake --build build --target $@ -- -s
	$(SUBMIT_CMD) nsys profile --force-overwrite true --output $@ ./build/$@



h01: h01.cpp
	cmake --preset system -B build -S . --log-level=WARNING
	cmake --build build --target $@ -- -s
	$(SUBMIT_CMD) ./build/$@

h01.nsys: h01.cpp
	cmake --preset system -B build -S . --log-level=WARNING
	cmake --build build --target $@ -- -s
	$(SUBMIT_CMD) nsys profile --force-overwrite true --output $@ ./build/$@



h02: h02.cpp
	cmake --preset system -B build -S . --log-level=WARNING
	cmake --build build --target $@ -- -s
	$(SUBMIT_CMD) ./build/$@

h02.nsys: h02.cpp
	cmake --preset system -B build -S . --log-level=WARNING
	cmake --build build --target $@ -- -s
	$(SUBMIT_CMD) nsys profile --force-overwrite true --output $@ ./build/$@