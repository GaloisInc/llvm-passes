LLVM_CONFIG = llvm-config-9

passes.so: Hello.o
	$(CXX) -fPIC -shared -o $@ $< `$(LLVM_CONFIG) --ldflags`

%.o: %.cpp
	$(CXX) -fPIC -fno-rtti -fno-exceptions -c -o $@ $< `$(LLVM_CONFIG) --cxxflags`

