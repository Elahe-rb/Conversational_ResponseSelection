from argsparams import args
import main_BERT

################ for hierarchical network ##################################
for cluster in range(args.num_clusters):
    print("Learning network::: [ " + str(cluster) + " ] started ...")
    args.network_num = cluster
    main_BERT.run(args)
