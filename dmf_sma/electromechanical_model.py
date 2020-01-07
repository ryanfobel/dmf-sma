"""
This module can be used to calculate and/or plot the electrical properties,
stored energy, and/or electromechanical forces acting on a drop within a
2-plate digital microfluidic device.

1. Chatterjee et al., "Electromechanical model for actuating liquids in a
   two-plate droplet microfluidic device," Lab on a Chip, no. 9
   (2009): 1219-1229.
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import sympy
import copy

epsilon_0 = 8.85e-12
default_frequencies = np.logspace(2,5,100)


def plot_impedance_magnitude_vs_frequency(electrode_models,
                                          frequencies=None,legend=True):
    """
    Plot the magnitude of the total impedance for different fluids.

    Parameters
    ----------
    electrode_models : [ElectrodeModel(...),...]
        List of ElectrodeModel objects.
    frequencies : np.array, optional
        List of frequencies (Hz) for which to plot the impedance. The default
        is np.logspace(2,5,100).
    legend : bool, optional
        A legend will be added to the figure if this parameter is True
        (default).

    Returns
    -------
    None

    Example
    -------
    >>> import dmf
    >>> e_list = [dmf.ElectrodeModel(name_fl="di water"),
    >>>           dmf.ElectrodeModel(name_fl="methanol"),
    >>>           dmf.ElectrodeModel(name_fl="ethanol"),
    >>>           dmf.ElectrodeModel(name_fl="acetone")]
    >>> dmf.plot_impedance_magnitude_vs_frequency(e_list)
    """
    if(frequencies is None):
        frequencies = default_frequencies
    if(type(electrode_models) != list):
        electrode_models = [electrode_models]
    plt.figure()
    legend_list = []
    for e in electrode_models:
        legend_list.append(e.parts["fluid"].name)
        plt.loglog(frequencies,abs(e.Z(frequencies)))
    a = plt.gca()
    a.set_xlim(frequencies[0],frequencies[-1])
    plt.title("Impedance magnitude")
    plt.ylabel("|Z| ($\Omega$)")
    plt.xlabel("Frequency (Hz)")
    if(legend):
        plt.legend(legend_list)

def plot_impedance_phase_vs_frequency(electrode_models,
                                      frequencies=None,
                                      legend=True):
    """
    Plot the phase of the total impedance for different fluids.

    Parameters
    ----------
    electrode_models : [ElectrodeModel(...),...]
        List of ElectrodeModel objects.
    frequencies : np.array, optional
        List of frequencies (Hz) for which to plot the impedance. The default
        is np.logspace(2,5,100).
    legend : bool, optional
        A legend will be added to the figure if this parameter is True
        (default).

    Returns
    -------
    None

    Example
    -------
    >>> import dmf
    >>> e_list = [dmf.ElectrodeModel(name_fl="di water"),
    >>>           dmf.ElectrodeModel(name_fl="methanol"),
    >>>           dmf.ElectrodeModel(name_fl="ethanol"),
    >>>           dmf.ElectrodeModel(name_fl="acetone")]
    >>> dmf.plot_impedance_phase_vs_frequency(e_list)
    """
    if(frequencies is None):
        frequencies = default_frequencies
    if(type(electrode_models) != list):
        electrode_models = [electrode_models]
    plt.figure()
    legend_list = []
    for e in electrode_models:
        legend_list.append(e.parts["fluid"].name)
        plt.semilogx(frequencies,np.angle(e.Z(frequencies), deg=True))
    a = plt.gca()
    a.set_xlim(frequencies[0],frequencies[-1])
    plt.title("Impdedance phase")
    plt.ylabel("Phase (degrees)")
    plt.xlabel("Frequency (Hz)")
    if(legend):
        plt.legend(legend_list)

def plot_normalized_force_vs_frequency(electrode_models,
                                       frequencies=None,
                                       legend=True):
    if(frequencies is None):
        frequencies = default_frequencies
    if(type(electrode_models) != list):
        electrode_models = [electrode_models]
    plt.figure()
    legend_list = []
    for e in electrode_models:
        legend_list.append(e.parts["fluid"].name)
        plt.semilogx(frequencies,e.normalized_force(frequencies))
    a = plt.gca()
    a.set_xlim(frequencies[0],frequencies[-1])
    plt.title("Normalized Force")
    plt.ylabel("Normalized Force (N/$V^2$)")
    plt.xlabel("Frequency (Hz)")
    if(legend):
        plt.legend(legend_list, loc="lower left")

def plot_relative_voltage_drop_vs_frequency(electrode_model,
                                            frequencies=None,
                                            legend=True):
    """
    Plot the relative voltage drop across each of the different device
    components as a function of frequency.

    Parameters
    ----------
    electrode_models : [ElectrodeModel(...),...]
        List of ElectrodeModel objects.
    frequencies : np.array, optional
        List of frequencies (Hz) for which to plot the relative voltage drop.
        The default is np.logspace(2,5,100).
    legend : bool, optional
        A legend will be added to the figure if this parameter is True
        (default).

    Returns
    -------
    None

    Example
    -------
    >>> import dmf
    >>> e = dmf.ElectrodeModel(name_fl="di water")
    >>> dmf.plot_relative_voltage_drop_vs_frequency(e)
    """
    if(frequencies is None):
        frequencies = default_frequencies
    plt.figure()
    legend_list = []
    for part in electrode_model.parts.keys():
        legend_list.append(electrode_model.parts[part].name)
        plt.semilogx(frequencies,
                     100*electrode_model.fractional_voltage_drop(part,
                     frequencies));
    plt.title("Relative voltage drop across device components (" + \
              electrode_model.fluid.name + ")")
    plt.ylabel("relative voltage drop (%)")
    plt.xlabel("Frequency (Hz)")
    if(legend):
        plt.legend(legend_list,loc="center left")

def plot_critical_frequency(electrode_models):
    """
    Plot the critical frequency for a list of ElectrodeModel objects.
    The default behaviour is to list objects by the name of the fluid.

    Parameters
    ----------
    electrode_models : [ElectrodeModel(...),...]
        List of ElectrodeModel objects.

    Returns
    -------
    None

    Example
    -------
    >>> import dmf
    >>> e_list = [dmf.ElectrodeModel(name_fl="di water"),
    >>>           dmf.ElectrodeModel(name_fl="methanol"),
    >>>           dmf.ElectrodeModel(name_fl="ethanol"),
    >>>           dmf.ElectrodeModel(name_fl="acetone")]
    >>> dmf.plot_critical_frequency(e_list)
    """
    ind = np.arange(np.size(electrode_models))
    height = 1
    legend_list = []
    cf = []
    for e in electrode_models:
        legend_list.append(e.parts["fluid"].name)
        cf.append(e.critical_frequency())
    plt.figure()
    plt.barh(ind, cf, log=True)
    plt.yticks(ind+height/2., legend_list)
    plt.xlabel("Critical Frequency (Hz)")
    plt.title("Critical frequency")

class ElectrodeModel(object):
    """
    Electromechanical model of a 2-plate DMF electrode.

    Instance Properties
    -----------------
    A : float
        Area of the electrode covered by fluid.
    fluid : Material object
        Fluid covering the electrode.
    hp_layer_top : Material object
        Top hydrophobic layer.
    hp_layer_bottom : Material object
        Bottom hydrophobic layer.
    de_layer : Material object
        Dielectric layer.

    Public Methods
    --------------
    Z() :
        Impedance of the entire device.
    critical_frequency() :
        Critical frequency of the device/fluid.
    device_capacitance():
        Combined capacitance of the dielectric and hydrophobic layers.
    fractional_voltage_drop() :
        Fraction of the voltage that drops across a given part.
    normalized_energy() :
        Total normalized energy stored in all all components of the device.
    normalized_force() :
        Total normalized force on the drop.
    """

    def __init__(self, A=1e-3**2,
                 name_fl="di water", d_fl=150e-6,
                 name_hp="teflon-af", d_hp=235e-9,
                 name_de="parylene-c", d_de=6e-6):
        """
        Create a 2-plate electrode model.

        Parameters
        ----------
        A : float, optional
            Surface area of the drop in meters squared (default is 1e-3**2).
        name_fl : str, optional
            Name of the fluid (default is "di water"). Other options can be listed
            using the command:
            >>> dmf.Material.properties.keys()
        d_fl : float, optional
            Gap height or distance between the top and bottom plate in meters
            (i.e. the height of the drop (default is 150e-6).
        name_hp : str, optional
            Name of the material used as a hydrophopic layer (default is
            "teflon-af").
        d_hp : float, optional
            Height or distance of the hydrophobic layer in meters (default is
            235e-9).
        name_de : float, options
            Name of the dielectric layer (default is "parylene-c").
        d_de : float, optional
            Height or distance of the dielectric layer in meters (default is 6e-6).
        """
        self.parts = {}
        self.hp_layer_top = Material(A, name_hp.lower(), d_hp)
        self.hp_layer_bottom = Material(A, name_hp.lower(), d_hp)
        self.fluid = Material(A, name_fl.lower(), d_fl)
        self.de_layer = Material(A, name_de.lower(), d_de)

        self.parts["hydrophobic layer (top)"] = self.hp_layer_top
        self.parts["hydrophobic layer (bottom)"] = self.hp_layer_top
        self.parts["fluid"] = self.fluid
        self.parts["dielectric layer"] = self.de_layer
        self.A = A

    @property
    def A(self):
        """Area of the electrode covered by fluid in m^2."""
        return self.parts["fluid"].A
    @A.setter
    def A(self, value):
        for p in self.parts.values():
            p.A = value

    def Z(self, freq=None):
        """
        Impedance of the entire device in Ohms.

        Parameters
        ----------
        freq : float or numpy.array, optional
            A single or array of frequencies for which to evaluate the
            impedance.

        Returns
        -------
        out : float, sympy.core.add.Add or a numpy.array of either
            Impedance in Ohms. If no frequency is supplied, a symbolic
            expression (sympy object) is returned as a function of the symbol
            f.
        """
        if freq is None:
            Z = 0
            for p in self.parts.values():
                Z = Z + p.Z()
            return Z
        else:
            if(np.size(freq)==1):
                Z = 0
            else:
                Z = np.zeros(np.size(freq));
            for p in self.parts.values():
                Z = Z + p.Z(freq)
            return Z

    def critical_frequency(self):
        "Critical frequency of the device/fluid (Hz)."
        return 1/(2*math.pi*self.parts["fluid"].R() * \
                  (self.parts["fluid"].C() + self.device_capacitance()))

    def device_capacitance(self):
        "Combined capacitance of the dielectric and hydrophobic layers (F)."
        return 1/(1/self.parts["hydrophobic layer (top)"].C() + \
        1/self.parts["hydrophobic layer (bottom)"].C() + \
        1/self.parts["dielectric layer"].C())

    def fractional_voltage_drop(self, part, freq=None):
        """
        Fraction of the voltage that drops across a given part.

        Parameters
        ----------
        part: str
            Part of the device for which you want to calculate the fractional
            voltage drop.  Possible values include: 'dielectric layer','fluid',
            'hydrophobic layer (bottom)', and 'hydrophobic layer (top)'.
        freq : float or numpy.array, optional
            A single or array of frequencies for which to evaluate the
            fractional voltage drop.

        Returns
        -------
        out : float, sympy.core.add.Add or a numpy.array of either
            Fractional voltage drop. If no frequency is supplied, a symbolic
            expression (sympy object) is returned as a function of the symbol
            f.
        """
        if freq is None:
            return abs(self.parts[part].Z()/self.Z())
        else:
            return abs(self.parts[part].Z(freq)/self.Z(freq))

    def normalized_energy(self, freq=None):
        """
        Total normalized energy stored in all all components of the device
        (J/V^2). Muliply this value by Vrms^2 to get the absolute energy (J).

        Parameters
        ----------
        freq : float or numpy.array, optional
            A single or array of frequencies for which to evaluate the
            normalized energy.

        Returns
        -------
        out : float, sympy.core.add.Add or a numpy.array of either
            Normalized energy. If no frequency is supplied, a symbolic
            expression (sympy object) is returned as a function of the symbol
            f.
        """
        if freq is None:
            U = 0
            for k, v in self.parts.items():
                if k != "feedback pot":
                    U = U + 0.5*v.C()*abs(v.Z()
                            /self.Z())**2
        else:
            if(np.size(freq)==1):
                U = 0
            else:
                U = np.zeros(np.size(freq));
            for k, v in self.parts.items():
                if k != "feedback pot":
                    U = U + 0.5*v.C()*abs(v.Z(freq)/
                            self.Z(freq))**2
        return U

    def normalized_force(self, freq=None, filler="air", length=None):
        """
        Normalized force on the drop (N/V^2). Muliply this value by Vrms^2 to
        get the absolute force (N).

        Parameters
        ----------
        freq : float or numpy.array, optional
            A single or array of frequencies for which to evaluate the
            normalized force.

        Returns
        -------
        out : float, sympy.core.add.Add or a numpy.array of either
            Normalized force. If no frequency is supplied, a symbolic
            expression (sympy object) is returned as a function of the symbol
            f.
        """
        if(length is None):
            length = math.sqrt(self.A)
        f = copy.deepcopy(self)
        f.fluid.name = filler
        return (self.normalized_energy(freq)-f.normalized_energy(freq))/ \
            (self.A/length)

class ElectricalProperties():
    def __init__(self,relative_permittivity,conductivity):
        self.relative_permittivity = relative_permittivity
        self.conductivity = conductivity

class Material():
    """
    Material properties.

    Public Methods
    --------------
    C :
        Capacitance of the material.
    R :
        Resistance of the material.
    Z :
        Impedance of the material.

    References:
    -----------
    1.  Chatterjee et al. Droplet-based microfluidics with nonaqueous
        solvents and solutions. Lab Chip 6, 199-206 (2006).
    2.  http://www.lenntech.com/teflon.htm
    3.  http://www.vp-scientific.com/parylene_properties.htm
    4.  Huang et al. Introducing dielectrophoresis as a new force field for
        field-flow fractionation. Biophysical Journal 73, 11188-1129 (1997).
    5.  http://www.amresco-inc.com/TE-BUFFER-PH-8.0-E112.cmsx
    6.  http://en.wikipedia.org/wiki/Purified_water
    7.  http://web.mit.edu/6.777/www/matprops/pdms.htm
    8.  Chatterjee et al. Electromechanical Model for Actuating Liquids
        in a Two-plate Droplet Microfluidic Device. Lab Chip 9, 1219-1229 (2009).
    9.  https://extranet.fisher.co.uk/webfiles/uk/web-docs/450_CH.pdf
    """
    properties = {"air":ElectricalProperties(1, 0),
                  "formamide":ElectricalProperties(111, 3.5e-3), # [1]
       	          "water":ElectricalProperties(80.1, 8.7e-4), # [1]
       	          "formic acid":ElectricalProperties(51.1, 7e-3), # [1]
       	          "dmso":ElectricalProperties(47.2, 3e-5), # [1]
       	          "dmf":ElectricalProperties(38.3, 3.2e-5), # [1]
       	          "acetonitrile":ElectricalProperties(36.6, 1.9e-5), # [1]
       	          "methanol":ElectricalProperties(33, 1.7e-4), # [1]
       	          "ethanol":ElectricalProperties(25.3, 7.4e-5), # [1]
       	          "acetone":ElectricalProperties(21, 5e-7), # [1]
       	          "piperidine":ElectricalProperties(4.3, 1e-5), # [1]
       	          "1-pentanol":ElectricalProperties(15.1, 8e-7), # [1]
       	          "1-hexanol":ElectricalProperties(13, 1.6e-5), # [1]
       	          "dichloromethane":ElectricalProperties(8.9, 1e-7), # [1]
       	          "dibromomethane":ElectricalProperties(7.8, 2.6e-6), # [1]
       	          "thf":ElectricalProperties(7.5, 5e-8), # [1]
       	          "chloroform":ElectricalProperties(4.8, 7e-8), # [1]
       	          "toluene":ElectricalProperties(2.4, 8e-14), # [1]
       	          "carbon tetrachloride":ElectricalProperties(2.2, 4e-16), # [1]
       	          "cyclohexane":ElectricalProperties(2, 7e-16), # [1]
       	          "teflon-af":ElectricalProperties(1.93, 0), # [2]
       	          "parylene-c":ElectricalProperties(3.10, 0), # [3]
       	          "cell media":ElectricalProperties(80, 1.5), # [4]
       	          "te buffer":ElectricalProperties(80, 8.75e-4), # [5]
       	          "di water":ElectricalProperties(80, 7.5e-5), # [6]
                  "polydimethylsiloxane":ElectricalProperties(2.6, 2.5e-14), # [7]
		          "30% glycerol":ElectricalProperties(58, 3e-4), # [8]
		          "60% glycerol":ElectricalProperties(51, 7e-5), # [8]
		          "90% glycerol":ElectricalProperties(43, 4e-6), # [8]
       	          "pbs":ElectricalProperties(80, 1.5), # [9]
                 }
    def __init__(self,A,name,d):
        """
        Create a new material.
        """
        self.name = name.lower()
        self.A = A
        self.d = d

    def __str__(self):
        if self.R() == float('inf'):
            return "%s, C=%.2e F" % (self.name, self.C())
        elif self.C() == float('inf'):
            return "%s, R=%.2e ohms" % (self.name, self.R())
        else:
            return "%s, C=%.2e F, R=%.2e ohms" % (self.name, self.C(), self.R())

    def relative_permittivity(self):
        return self.__class__.properties[self.name].relative_permittivity

    def resistivity(self):
        conductivity = self.__class__.properties[self.name].conductivity
        if conductivity == 0:
            return float('inf')
        return 1/conductivity

    def C(self):
        "Capacitance (F)."
        return epsilon_0*self.relative_permittivity()*self.A/self.d

    def R(self):
        "Resistance (Ohms)."
        return self.resistivity()*self.d/self.A

    def Z(self, freq=None):
        "Resistance (Ohms)."
        if freq is None:
            f = sympy.Symbol('f')
            if self.resistivity() == float('inf'):
                return -sympy.I/(2*math.pi*self.C()*f)
            return self.R()/(1 + 2*math.pi*sympy.I*self.R()*self.C()*f)
        else:
            if self.resistivity() == float('inf'):
                return -1j/(2*math.pi*self.C()*freq)
            return self.R()/(1 + 2*math.pi*1j*self.R()*self.C()*freq)

class Resistor():
    def __init__(self,resistance):
        self.resistance = resistance

    def C(self):
        return float('inf')

    def R(self):
        return self.resistance

    def Z(self, freq=None):
        return self.R()

class Capacitor():
    def __init__(self,capacitance):
        self.capacitance = capacitance

    def C(self):
        return self.capacitance

    def R(self):
        return float('inf')

    def Z(self, freq=None):
        if freq is None:
            f = sympy.Symbol('f')
            return 1/(2*math.pi*1j*self.C()*f)
        else:
            return 1/(2*math.pi*1j*self.C()*freq)


if __name__ ==  "__main__":
    f_start = 1e2
    f_end = 1e6

    # Adjust these values.
    A=2.5e-3**2
    d_fl=280e-6
    name_hp="Teflon-AF"
    d_hp=235e-9  #235e-9
    name_de="Parylene-C"
    d_de= 6e-6 #2.2e-6 #6e-6

    # These are the fluids that will be used in the simulations. Uncomment any
    # fluids that you'd like to add.
    fluids = [
    # dielectrics:
    #          "Air",
    #          "Polydimethylsiloxane",
    #          "Carbon tetrachloride",
    #          "Cyclohexane",
             "Toluene",
    #
    # "conductive" liquids:
    #          "Formamide",
             "DI water",
    #          "Water",
    #          "Formic acid",
    #          "DMSO",
    #          "DMF",
    #          "Acetonitrile",
             "Methanol",
    #          "Ethanol",
    #          "Acetone",
    #          "Piperidine",
    #          "1-Pentanol",
    #          "1-Hexanol",
    #          "Dichloromethane",
    #          "Dibromomethane",
    #          "THF",
    #          "Chloroform",
    #          "Cell media",
    #          "TE buffer",
    #          "30% glycerol",
    #          "60% glycerol",
    #          "90% glycerol",
             "PBS",
             ]

    # Print out the simulation parameters.
    freq = 10**np.linspace(math.log10(f_start),math.log10(f_end),100)
    print("Simulation parameters")
    print("======================================================")
    print("Electrode area: %.1f mm^2" % (A*1e6))
    print("Gap height: %.1f um" % (d_fl*1e6))
    print("Hydrophobic layer (x2): " + str(d_hp*1e9) + " nm " \
          + str(Material(A,name_hp,d_hp)))
    print("Dielectric layer: " + str(d_de*1e6) + " um " \
          + str(Material(A,name_de,d_de)))
    e = ElectrodeModel(A=A, name_hp=name_hp, d_hp=d_hp, \
                       name_de=name_de, d_de=d_de, \
                       name_fl="Water", d_fl=d_fl)
    print("Device capacitance = %.2e F\n" % e.device_capacitance())

    f = sympy.Symbol('f')

    # Print the Device impedance values.
    CF = []
    for v in fluids:
        e = ElectrodeModel(A=A, name_hp=name_hp, d_hp=d_hp, \
                           name_de=name_de, d_de=d_de, \
                           name_fl=v, d_fl=d_fl)
        print("Fluid: " + e.parts["fluid"].__str__())
        print("Critical Frequency = %.1e Hz" % e.critical_frequency())
        CF.append(e.critical_frequency())
        print("Z_total = " + str(sympy.simplify(e.Z())))
        print("|Z_total| = " + str(sympy.sqrt(sympy.simplify(
                             (sympy.conjugate(e.Z())*e.Z())
                             .subs({sympy.conjugate(f):f})))))
        R_L = e.parts['fluid'].R()
        C_L = e.parts['fluid'].C()
        C_D = 1/(1/e.parts['dielectric layer'].C() + \
              1/e.parts['hydrophobic layer (top)'].C() + \
              1/e.parts['hydrophobic layer (bottom)'].C())
        print("R_L*(C_L+C_D) = %.1e" % (R_L*(C_L+C_D)))
        print("C_D = %.1e" % (C_D))
        print("R_L*C_L*C_D = %.1e\n" % (R_L*C_L*C_D))

    # Plot the reltive voltage drop for each device component (with water
    # as the fluid).
    e = ElectrodeModel(A=A, name_hp=name_hp, d_hp=d_hp, \
                       name_de=name_de, d_de=d_de, \
                       name_fl="Water", d_fl=d_fl)
    plot_relative_voltage_drop_vs_frequency(e, freq)

    # Plot the reltive voltage drop for each device component
    e = ElectrodeModel(A=A, name_hp=name_hp, d_hp=d_hp, \
                       name_de=name_de, d_de=d_de, \
                       name_fl="30% glycerol", d_fl=d_fl)
    plot_relative_voltage_drop_vs_frequency(e, freq)

    # Plot the reltive voltage drop for each device component
    e = ElectrodeModel(A=A, name_hp=name_hp, d_hp=d_hp, \
                       name_de=name_de, d_de=d_de, \
                       name_fl="60% glycerol", d_fl=d_fl)
    plot_relative_voltage_drop_vs_frequency(e, freq)

    # Plot the critical frequency and impedance values for each of the
    # different fluids.
    e_list = []
    for v in fluids:
        e_list.append(ElectrodeModel(A=A, name_hp=name_hp, d_hp=d_hp, \
                                         name_de=name_de, d_de=d_de, \
                                         name_fl=v, d_fl=d_fl))
    #plot_critical_frequency(e_list)
    plot_impedance_magnitude_vs_frequency(e_list, freq)
    #plot_impedance_phase_vs_frequency(e_list, freq)
    #plot_normalized_force_vs_frequency(e_list, freq)

    conductivity = np.logspace(-8,1,100)
    relative_permativitiy = (1, 10, 100)
    frequency = 1e3
    e = ElectrodeModel(A=A, name_hp=name_hp, d_hp=d_hp, \
                       name_de=name_de, d_de=d_de, \
                       d_fl=d_fl)

    """
    plt.figure()
    legend_list = []
    for p in relative_permativitiy:
      force = []
      legend_list.append("%d" % p)
      for c in conductivity:
          name = "%.1e" % c
          Material.properties[name] = ElectricalProperties(p, c)
          e.fluid.name = name
          force.append(1e6*e.normalized_force(frequency)*200**2)
      plt.semilogx(conductivity, force)
    plt.legend(legend_list, loc="lower right")
    plt.xlabel("Conductivity (S/m)")
    plt.ylabel("Driving force ($\mu$N)")
    plt.title("%d $\mu$m Parylene-C, 200 V$_{RMS}$" % (d_de*1e6))
    """

    plt.figure(dpi=150, facecolor="white")
    legend_list = []
    for e in e_list:
        legend_list.append(e.parts["fluid"].name)
        plt.semilogx(freq,1e6*e.normalized_force(freq)*100**2)
    a = plt.gca()
    a.set_xlim(freq[0],freq[-1])
    plt.title("Driving force at 100 V$_{VRMS}$")
    plt.ylabel("Force ($\mu$N)")
    plt.xlabel("Frequency (Hz)")
    plt.legend(legend_list, loc="center left")

    plt.show()
