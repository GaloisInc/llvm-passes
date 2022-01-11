LLVM_CONFIG = llvm-config-9

OBJS = \
			 Hello.o \
			 CcInstrument.o \
			 UnrollVectors.o \
			 SoftFloat.o \
			 SetIntrinsicAttrs.o \
			 FlattenInit.o

passes.so: $(OBJS)
	$(CXX) $(CXXFLAGS) -ggdb -fPIC -shared -o $@ $^ `$(LLVM_CONFIG) --ldflags`

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -ggdb -fPIC -fno-rtti -fno-exceptions -c -o $@ $< `$(LLVM_CONFIG) --cxxflags`

