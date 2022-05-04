##################################################################
# Import machine specific compilers, options, flags, etc.. 
##################################################################

BRAID_DIR=#path to xbraid
include #path to /makefile.inc


##################################################################
# Build
##################################################################

HYPRE_DIR = #path to hypre
HYPRE_FLAGS = -I$(HYPRE_DIR)/include
HYPRE_LIB = -L$(HYPRE_DIR)/lib -lHYPRE
HYPRE_LIB_FILE = $(HYPRE_DIR)/lib/libHYPRE.a

#MFEM_DIR = ../../mfem
#MFEM_CONFIG_MK = $(MFEM_DIR)/config/config.mk
#MFEM_LIB_FILE = mfem_is_not_built
#-include $(MFEM_CONFIG_MK)

BRAID_FLAGS = -I$(BRAID_DIR)
BRAID_LIB_FILE = $(BRAID_DIR)/libbraid.a

C_NOHYPRE = xbraid
#CPP_NOHYPRE = ex-01-pp
#F_NOHYPRE = ex-01-expanded-f
#C_EXAMPLES = ex-03 ex-03-serial
# Note: .cpp examples will be linked with mfem
#CXX_EXAMPLES = ex-04

.PHONY: all clean cleanout

.SUFFIXES:
.SUFFIXES: .c .cpp

# put this rule first so it becomes the default
all: $(C_NOHYPRE) $(CPP_NOHYPRE) $(C_EXAMPLES) $(CXX_EXAMPLES)


# Rule for building ex-02
xbraid: xbraid.c $(BRAID_LIB_FILE)
	@echo "Building" $@ "..."
	$(MPICC) $(CFLAGS) $(BRAID_FLAGS) $(@).c -o $@ $(BRAID_LIB_FILE) $(LFLAGS)

clean: cleanout
	rm -f *.o $(C_NOHYPRE) $(CPP_NOHYPRE) $(F_NOHYPRE) $(C_EXAMPLES) $(CXX_EXAMPLES) $(F_EXAMPLES) *ror_norm* *_err_* *_mesh* *_sol_*
	rm -rf *.dSYM

cleanout:
	rm -f ex*.out.*

