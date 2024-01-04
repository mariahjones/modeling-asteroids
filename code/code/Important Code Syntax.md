to change file name:
    shift>command>p

saving pickle file
    import pickle
    file = open('filename.pkl', 'wb')
    pickle.dump(ast,file)
    file.close()

loading in pickle file
    import pickle
    file = open('filename.pkl', 'wb')
    pickle.dump(ast,file)
    file.close()

backup saving files
    np.savez(
        'filename.npz',
        temps=ast.temps, asteroid=ast.asteroid, epoch=ast.epoch,
        thicknesses=ast.thicknesses, depths=ast.depths, visible=ast.visible,
        cond=ast.cond, therm_inert=ast.therm_inert
        )

to edit ssh config file:
    nano ~/.ssh/config
to save
    control+O, enter, control+X
    

to connect to scott/amundsen:
    username : mariahjones@scott.grid.uchicago.edu
    password : JupiterRuns27!

to see all directories:
    ls -a

to update spt3g sotfware:
    cd /home/mariahjones/code/spt3g_software
    git pull --rebase
    # you’ll need to enter your username and the long github token that has all the annoying characters
    rm -r build
    mkdir build
    cd build
    cmake ../
    make -j 12

to save data set as numpy array
    np.savez( ‘file_name.npz’ , name of created list = np.array(name of created list) )

to load data from saved file
    var = np.load(’file_name.npz’)[name of created list] #loads in dictionary, index list

to combine numpy lists
    var = np.concatenate([list1, list2,…])


to copy from directory to home directory
    cp -r ‘directory to copy’ ‘directory to place copy’

for dark jupyter theme, setting plot background
    import matplotlib as plt
    plt.style.use(‘dark_background’)

plotting parameters
    fig = plt.figure()
    ax = fig.add_subplot()

    ax.set_facecolor(‘color’)
    ax.xaxis.label.set_color(‘collor’)        
    ax.yaxis.label.set_color(‘color’)          

    ax.tick_params(axis='x', colors=‘color’)    
    ax.tick_params(axis='y', colors=‘color’)  

    ax.spines['left'].set_color(‘color’)        
    ax.spines['top'].set_color(‘color’)         

    plt.title('your_title’, fontweight = 'bold' , color = ‘color’)
    plt.xlabel('your_label', fontweight='bold', color = ‘color’)
    plt.ylabel(‘your_label’, fontweight='bold', color = ‘color’)

    plt.xlim(-x,x)
    plt.ylim(-y,y)

    plt.plot(data)
    plt.savefig(“file_name.jpeg")
    plt.show()

subplotting parameters
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,20))

    for i, temps in enumerate(subsolar_temps):
        if i != 0:
            pct_change = ((subsolar_temps[i] - subsolar_temps[i-1])/subsolar_temps[i-1])
        ax[0].bar(i, height = pct_change , label = “your_label”.format(i))
        ax[0].set_title('your_title’, color = ‘color’, fontweight = 'bold')
        ax[1].set_title('your_title’, color = ‘color’, fontweight = 'bold')
    ax[1].plot(subsolar_temps.T)
    ax[0].set_xlabel(“your_label”, color = ‘color’, fontweight = 'bold')
    ax[0].set_ylabel(“your_label”, color = ‘color’, fontweight = 'bold')
    ax[1].set_xlabel(“your_label”, color = ‘color’, fontweight = 'bold')
    ax[1].set_ylabel(“your_label”, color = ‘color’, fontweight = 'bold')
    ax[0].legend()

    plt.show()
        
