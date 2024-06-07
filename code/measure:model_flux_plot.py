# create pandas dataframe
# pre-define the frequencies to make code easier later
freq_dict = {
    'JCMT': [857* core.G3Units.GHz, 666* core.G3Units.GHz, 375* core.G3Units.GHz, 273* core.G3Units.GHz, 231* core.G3Units.GHz],
    'IRAS': [24983* core.G3Units.GHz, 11992* core.G3Units.GHz, 4997* core.G3Units.GHz, 2998* core.G3Units.GHz],
    'SPT': [90* core.G3Units.GHz, 150* core.G3Units.GHz, 220* core.G3Units.GHz],
    'VLA': [15* core.G3Units.GHz]
}

flux_dict = {
    'JCMT': [13.1* core.G3Units.Jy, 8.47* core.G3Units.Jy, 2.62* core.G3Units.Jy, 1.42* core.G3Units.Jy, 1.15* core.G3Units.Jy],
    'IRAS': [57.7* core.G3Units.Jy, 111.7* core.G3Units.Jy, 50.3* core.G3Units.Jy, 20.3* core.G3Units.Jy],
    'SPT': [155.84* core.G3Units.mJy, 431.77* core.G3Units.mJy, 1018.89* core.G3Units.mJy],
    'VLA': [0.001085 * core.G3Units.Jy]
}

noise_dict = {
    'JCMT': [1.3* core.G3Units.Jy, 0.85* core.G3Units.Jy, 0.26* core.G3Units.Jy, 0.14* core.G3Units.Jy, 0.12* core.G3Units.Jy],
    'IRAS': [6.2* core.G3Units.Jy, 12.3* core.G3Units.Jy, 16.5* core.G3Units.Jy, 5.2* core.G3Units.Jy],
    'SPT': [2.07* core.G3Units.mJy, 3.09* core.G3Units.mJy, 7.76* core.G3Units.mJy],
    'VLA': [0.000036 * core.G3Units.Jy]
}


# empty dataframe that we'll store our outputs in
df = pd.DataFrame()

for thermal_inertia in [5, 20, 80]:
    for telescope in ['JCMT', 'IRAS', 'SPT', 'VLA']:

        # read in asteroid model
        filename = '/home/mariahjones/{}_Data{}.pkl'.format(telescope, thermal_inertia)
        file = open(filename, 'rb')
        ast = pickle.load(file)
        file.close()

        # get fluxes at freqs we care about for that telescope
        for freq, mes, noise in zip(freq_dict[telescope], flux_dict[telescope], noise_dict[telescope]):
            flux = asm.flux(ast, freq)
            wavelength = (c * core.G3Units.m / core.G3Units.s) / freq

            # save our info in the dataframe
            df = df.append({
                'telescope': telescope,
                'thermal_inertia': thermal_inertia,
                'wavelength': wavelength,
                'frequency': freq,
                'flux': flux,
                'measure': mes,
                'noise':noise
            }, ignore_index=True).reset_index(drop=True)

#plot dataframe
w = (df['thermal_inertia'] == 5) * (df['telescope'] == "IRAS")
plt.errorbar(x=df[w]["wavelength"] / core.G3Units.mm, 
             y=df[w]["measure"] / df[w]["flux"], 
             yerr = df[w]["noise"]/ df[w]["flux"], 
             fmt = "o",
            c = 'violet',
            label = "IRAS"
            )

w = (df['thermal_inertia'] == 5) * (df['telescope'] == "JCMT")
plt.errorbar(x=df[w]["wavelength"] / core.G3Units.mm, 
             y=df[w]["measure"] / df[w]["flux"], 
             yerr = df[w]["noise"]/ df[w]["flux"], 
             fmt = "o",
            c = 'aqua',
            label = "JCMT"
            )
w = (df['thermal_inertia'] == 5) * (df['telescope'] == "SPT")
plt.errorbar(x=df[w]["wavelength"] / core.G3Units.mm, 
             y=df[w]["measure"] / df[w]["flux"], 
             yerr = df[w]["noise"]/ df[w]["flux"], 
             fmt = "o",
            c = 'pink',
            label = "SPT"
            )

w = (df['thermal_inertia'] == 5) * (df['telescope'] == "VLA")
plt.errorbar(x=df[w]["wavelength"] / core.G3Units.mm, 
             y=df[w]["measure"] / df[w]["flux"], 
             yerr = df[w]["noise"]/ df[w]["flux"], 
             fmt = "o",
            c = 'orange',
            label = "VLA"
            )
plt.axhline(1)
plt.xscale("log")
plt.title("Pallas Flux vs Wavelength" "\n" r"thermal inertia = $5 J / (Km^2 s^{0.5})$")
plt.xlabel("Wavelength (mm)")
plt.ylabel("Measured/Modeled Flux")
plt.grid(alpha=0.4)
plt.ylim(.4,1.3)
plt.legend()
#plt.savefig("Pallas Flux 5.jpeg")
plt.show()