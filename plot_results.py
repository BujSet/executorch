import matplotlib.pyplot as plt

# plot operator binary sizes
with open("results.txt", "r") as rf:
    lines = rf.readlines()
    operatorResultsHeaders = lines[0].split(",")
    operatorResults = [line for line in lines if "aten::" in line] 
    opResults = []
    for opResult in operatorResults:
        values = opResult.split(",")
        name = values[0].replace("aten::", "").replace(".out", "").replace("_out", "").strip()
        binSize = int(values[1].strip())
        opResults.append( (name, binSize) )

         
    opResults.sort()
    opNames = [name for (name, binSize) in opResults]
    opNames = ["_native_batch_norm_legit\n_no_training" if "native_batch_norm" in name else name for name in opNames] 
    opBinSizes = [binSize for (name, binSize) in opResults]

    fig = plt.figure(figsize=(18, 6))
    plt.grid(axis='y', zorder=0)
    plt.bar(opNames, opBinSizes, width=0.9, zorder=3)
    plt.margins(x=0.01)
    plt.xlabel('Operator')
    plt.xticks([i for i in range(len(opNames))], opNames, rotation=85)
    plt.ylabel('Stripped Binary Size for Single\nOperator Selective Build (KB)')
    plt.yticks([i for i in range(0,500000,100000)], [str(int(i/1000)) for i in range(0,500000, 100000)], rotation=45)
    plt.title('Executorch Operator Selective Build')
    plt.tight_layout()
    plt.savefig('operator_binary_sizes.pdf', format='pdf')
    plt.close()

with open("results.txt", "r") as rf:
    lines = rf.readlines()
    lines = [line for line in lines if "aten::" not in line]
    lines = [line for line in lines if "OperatorName,StrippedBinarySize,CompilationTime(sec)" not in line]
    allOpResults = [line for line in lines if len(line.split(","))==6 and line.split(",")[2].strip()=="True"]
    allOpResults = [(line.split(",")[0], int(line.split(",")[4])) for line in allOpResults]
    opSelectResults = [line for line in lines if len(line.split(","))==6 and line.split(",")[2].strip()=="False" and line.split(",")[3].strip()=="OFF"]
    opSelectResults = [(line.split(",")[0], int(line.split(",")[4])) for line in opSelectResults]
    dtypeSelectResults = [line for line in lines if len(line.split(","))>=6 and line.split(",")[2].strip()=="False" and line.split(",")[3].strip()=="ON"]
    dtypeSelectResults = [(line.split(",")[0], int(line.split(",")[4])) for line in dtypeSelectResults]
    
    modelNames = sorted([name for (name, binSize) in allOpResults])
    baselines = [1.0 for _ in modelNames]
    opReduction = []
    dtypeReduction = []
    for model in modelNames:
        baselineSize = 0
        opSelSize = 0
        dtypeSelSize = 0

        for (searchModel, searchSize) in allOpResults:
            if model == searchModel:
                baselineSize = searchSize
        assert(baselineSize > 0)
        for (searchModel, searchSize) in opSelectResults:
            if model == searchModel:
                opSelSize = searchSize
        assert(opSelSize > 0)
        for (searchModel, searchSize) in dtypeSelectResults:
            if model == searchModel:
                dtypeSelSize = searchSize
        assert(dtypeSelSize > 0)
        opReduction.append(float(opSelSize)/float(baselineSize))
        dtypeReduction.append(float(dtypeSelSize)/float(baselineSize))
    fig, ax = plt.subplots(figsize=(18,6))
    plt.grid(axis='y', zorder=0)
    w = 1.0 / 3
    ax.bar([i-w for i in range(len(modelNames))], baselines, width=w, zorder=3, label="All Operators", edgecolor='black', hatch="/")
    ax.bar([i-0.00 for i in range(len(modelNames))], opReduction, width=w, zorder=3, label="Operator Selective Build", edgecolor='black', hatch="\\")
    ax.bar([i+w for i in range(len(modelNames))], dtypeReduction, width=w, zorder=3, label="Dtype Selective Build", edgecolor='black', hatch="x")
    ax.set_xlabel('Model')
    ax.set_xticks([i for i in range(len(modelNames))], modelNames, rotation=65)
    ax.set_ylabel('Relative Binary Size')
    #ax.set_yticks([i for i in range(0,500000,100000)], [str(int(i/1000)) for i in range(0,500000, 100000)], rotation=45)
    ax.legend(loc='upper right')
    ax.set_title("Build Size Reduction with Dtype Selection")
    plt.margins(x=0.01)
    plt.tight_layout()
    plt.savefig('dtype_selective_build.pdf', format='pdf')
    plt.close()

# Plot compilation times
with open("results.txt", "r") as rf:
    lines = rf.readlines()
    lines = [line for line in lines if "aten::" not in line]
    lines = [line for line in lines if "OperatorName,StrippedBinarySize,CompilationTime(sec)" not in line]
    allOpResults = [line for line in lines if len(line.split(","))==6 and line.split(",")[2].strip()=="True"]
    allOpResults = [(line.split(",")[0], float(line.split(",")[5])) for line in allOpResults]
    opSelectResults = [line for line in lines if len(line.split(","))==6 and line.split(",")[2].strip()=="False" and line.split(",")[3].strip()=="OFF"]
    opSelectResults = [(line.split(",")[0], float(line.split(",")[5])) for line in opSelectResults]
    dtypeSelectResults = [line for line in lines if len(line.split(","))>=6 and line.split(",")[2].strip()=="False" and line.split(",")[3].strip()=="ON"]
    dtypeSelectResults = [(line.split(",")[0], float(line.split(",")[5])) for line in dtypeSelectResults]
    
    modelNames = sorted([name for (name, binSize) in allOpResults])
    baselines = [1.0 for _ in modelNames]
    opReduction = []
    dtypeReduction = []
    for model in modelNames:
        baselineSize = 0
        opSelSize = 0
        dtypeSelSize = 0

        for (searchModel, searchSize) in allOpResults:
            if model == searchModel:
                baselineSize = searchSize
        assert(baselineSize > 0)
        for (searchModel, searchSize) in opSelectResults:
            if model == searchModel:
                opSelSize = searchSize
        assert(opSelSize > 0)
        for (searchModel, searchSize) in dtypeSelectResults:
            if model == searchModel:
                dtypeSelSize = searchSize
        assert(dtypeSelSize > 0)
        opReduction.append(float(opSelSize)/float(baselineSize))
        dtypeReduction.append(float(dtypeSelSize)/float(baselineSize))
    fig, ax = plt.subplots(figsize=(18,6))
    plt.grid(axis='y', zorder=0)
    w = 1.0 / 3
    ax.bar([i-w for i in range(len(modelNames))], baselines, width=w, zorder=3, label="All Operators", edgecolor='black', hatch="/")
    ax.bar([i-0.00 for i in range(len(modelNames))], opReduction, width=w, zorder=3, label="Operator Selective Build", edgecolor='black', hatch="\\")
    ax.bar([i+w for i in range(len(modelNames))], dtypeReduction, width=w, zorder=3, label="Dtype Selective Build", edgecolor='black', hatch="x")
    ax.set_xlabel('Model')
    ax.set_xticks([i for i in range(len(modelNames))], modelNames, rotation=65)
    ax.set_ylabel('Relative Compilation Time')
    #ax.set_yticks([i for i in range(0,500000,100000)], [str(int(i/1000)) for i in range(0,500000, 100000)], rotation=45)
    ax.legend(loc='upper right')
    ax.set_title("Compilation Overhead")
    plt.margins(x=0.01)
    plt.tight_layout()
    plt.savefig('dtype_selective_build_time.pdf', format='pdf')
    plt.close()

with open("results.txt", "r") as rf:
    lines = rf.readlines()
    lines = [line for line in lines if "aten::" not in line]
    lines = [line for line in lines if "OperatorName,StrippedBinarySize,CompilationTime(sec)" not in line]
    allOpResults = [line for line in lines if len(line.split(","))==6 and line.split(",")[2].strip()=="True"]
    allOpResults = [(line.split(",")[0], int(line.split(",")[4])) for line in allOpResults]
    opSelectResults = [line for line in lines if len(line.split(","))==6 and line.split(",")[2].strip()=="False" and line.split(",")[3].strip()=="OFF"]
    opSelectResults = [(line.split(",")[0], int(line.split(",")[4])) for line in opSelectResults]
    dtypeSelectResults = [line for line in lines if len(line.split(","))>=6 and line.split(",")[2].strip()=="False" and line.split(",")[3].strip()=="ON"]
    dtypeSelectResults = [(line.split(",")[0], int(line.split(",")[4])) for line in dtypeSelectResults]
    
    modelNames = sorted([name for (name, binSize) in allOpResults])
    baselines = [1.0 for _ in modelNames]
    opReduction = []
    dtypeReduction = []
    for model in modelNames:
        baselineSize = 0
        opSelSize = 0
        dtypeSelSize = 0

        for (searchModel, searchSize) in allOpResults:
            if model == searchModel:
                baselineSize = searchSize
        assert(baselineSize > 0)
        for (searchModel, searchSize) in opSelectResults:
            if model == searchModel:
                opSelSize = searchSize
        assert(opSelSize > 0)
        for (searchModel, searchSize) in dtypeSelectResults:
            if model == searchModel:
                dtypeSelSize = searchSize
        assert(dtypeSelSize > 0)
        opReduction.append(float(opSelSize)/float(baselineSize))
        dtypeReduction.append(float(dtypeSelSize)/float(baselineSize))
    fig, ax = plt.subplots(figsize=(18,6))
    plt.grid(axis='y', zorder=0)
    w = 1.0 / 3
    ax.bar([i-w for i in range(len(modelNames))], baselines, width=w, zorder=3, label="All Operators", edgecolor='black', hatch="/")
    ax.bar([i-0.00 for i in range(len(modelNames))], opReduction, width=w, zorder=3, label="Operator Selective Build", edgecolor='black', hatch="\\")
    ax.bar([i+w for i in range(len(modelNames))], dtypeReduction, width=w, zorder=3, label="Dtype Selective Build", edgecolor='black', hatch="x")
    ax.set_xlabel('Model')
    ax.set_xticks([i for i in range(len(modelNames))], modelNames, rotation=65)
    ax.set_ylabel('Relative Binary Size')
    #ax.set_yticks([i for i in range(0,500000,100000)], [str(int(i/1000)) for i in range(0,500000, 100000)], rotation=45)
    ax.legend(loc='upper right')
    ax.set_title("Build Size Reduction with Dtype Selection")
    plt.margins(x=0.01)
    plt.tight_layout()
    plt.savefig('dtype_selective_build.pdf', format='pdf')
    plt.close()

with open("results.txt", "r") as rf:
    lines = rf.readlines()
    lines = [line for line in lines if "aten::" not in line]
    lines = [line for line in lines if "OperatorName,StrippedBinarySize,CompilationTime(sec)" not in line]
    dtypeSel1Dtype = [line for line in lines if len(line.split(","))>=6 and line.split(",")[2].strip()=="False" and line.split(",")[3].strip()=="ON"]
    dtypeSel1Dtype = [(line.split(",")[0], int(line.split(",")[7])) for line in dtypeSel1Dtype]
    dtypeSel2Dtype = [line for line in lines if len(line.split(","))>=6 and line.split(",")[2].strip()=="False" and line.split(",")[3].strip()=="ON"]
    dtypeSel2Dtype = [(line.split(",")[0], int(line.split(",")[8])) for line in dtypeSel2Dtype]
    modelNames = sorted([name for (name, binSize) in dtypeSel1Dtype])
    oneDtype = []
    twoDtypes = []
    for model in modelNames:
        oneDtypeSel = 0
        twoDtypeSel = 0

        for (searchModel, searchSize) in dtypeSel1Dtype:
            if model == searchModel:
                oneDtypeSel = searchSize
        assert(oneDtypeSel > 0)
        for (searchModel, searchSize) in dtypeSel2Dtype:
            if model == searchModel:
                twoDtypeSel = searchSize
        assert(opSelSize >= 0)
       
        oneDtype.append(float(oneDtypeSel)/float(oneDtypeSel+twoDtypeSel))
        twoDtypes.append(float(twoDtypeSel)/float(oneDtypeSel+twoDtypeSel))
    fig, ax = plt.subplots(figsize=(18,6))
    plt.grid(axis='y', zorder=0)
    ax.bar([i for i in range(len(modelNames))], oneDtype, zorder=3, label="Single Dtype Used", edgecolor='black', hatch="\\")
    ax.bar([i for i in range(len(modelNames))], twoDtypes, zorder=3, bottom=oneDtype, label="Two Dtypes Used", edgecolor='black', hatch="/")
    ax.set_xlabel('Model')
    ax.set_xticks([i for i in range(len(modelNames))], modelNames, rotation=65)
    ax.set_ylabel('Fraction  of All Operators\nUsed in Model')
    ax.legend(loc='upper right')
    ax.set_title("Low Data Dtype Diversity Across All Operators")
    plt.margins(x=0.01)
    plt.tight_layout()
    plt.savefig('dtype_diversity.pdf', format='pdf')
    plt.close()

with open("results.txt", "r") as rf:
    lines = rf.readlines()
    lines = [line for line in lines if "aten::" not in line]
    lines = [line for line in lines if "OperatorName,StrippedBinarySize,CompilationTime(sec)" not in line]
    numOpResults = [line for line in lines if len(line.split(","))>=6 and line.split(",")[2].strip()=="False" and line.split(",")[3].strip()=="ON"]
    numOpResults = [(line.split(",")[0], int(line.split(",")[6])) for line in numOpResults]
    modelNames = sorted([name for (name, binSize) in numOpResults])
    numOps = []

    for model in modelNames:
        numOp = 0

        for (searchModel, searchSize) in numOpResults:
            if model == searchModel:
                numOp = searchSize
        assert(numOp > 0)
       
        numOps.append(numOp)
    fig, ax = plt.subplots(figsize=(18,6))
    plt.grid(axis='y', zorder=0)
    ax.bar([i for i in range(len(modelNames))], numOps, zorder=3)
    ax.set_xlabel('Model')
    ax.set_xticks([i for i in range(len(modelNames))], modelNames, rotation=65)
    ax.set_ylabel('Number of Operators Used in Model')
    ax.set_title("High Operator Diversity Across Models")
    plt.margins(x=0.01)
    plt.tight_layout()
    plt.savefig('ops_per_model.pdf', format='pdf')
    plt.close()
