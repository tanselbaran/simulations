import neuron
import LFPy
import os
from glob import glob

def get_templatename(f):
    '''
    Assess from hoc file the templatename being specified within

    Arguments
    ---------
    f : file, mode 'r'

    Returns
    -------
    templatename : str

    '''
    for line in f.readlines():
        if 'begintemplate' in line.split():
            templatename = line.split()[-1]
            print('template {} found!'.format(templatename))
            continue

    return templatename

def load_mechanisms_from_neuron_model(cell_name):
    neuron_model = '/home/baran/Desktop/neuron_models/'+cell_name
    cwd = os.getcwd()
    mechanisms_folder = neuron_model + '/mechanisms'
    os.chdir(mechanisms_folder)
    os.system('nrnivmodl') #Compiling mechanisms
    os.chdir(cwd)
    neuron.load_mechanisms(mechanisms_folder)

def load_cell_properties(cell_name):
    neuron_model = '/home/baran/Desktop/neuron_models/' + cell_name
    os.chdir(neuron_model)

    #get the template name
    f = open("template.hoc", 'r')
    templatename = get_templatename(f)
    f.close()

    #get biophysics template name
    f = open("biophysics.hoc", 'r')
    biophysics = get_templatename(f)
    f.close()

    #get morphology template name
    f = open("morphology.hoc", 'r')
    morphology = get_templatename(f)
    f.close()

    #get synapses template name
    f = open(os.path.join("synapses", "synapses.hoc"), 'r')
    synapses = get_templatename(f)
    f.close()

    #Loading physical constants and other things
    neuron.h.load_file('constants.hoc')

    #If morphology of the cell is not loaded, load it
    if not hasattr(neuron.h, morphology):
        neuron.h.load_file(1, "morphology.hoc")

    #If biophysics of the cell is not loaded, load it
    if not hasattr(neuron.h, biophysics):
        neuron.h.load_file(1, "biophysics.hoc")

    #If synapses of the cell are not loaded ,load it
    if not hasattr(neuron.h, synapses):
        neuron.h.load_file(1, os.path.join('synapses', "synapses.hoc"))

    #If not the main cell template is not loaded, load it
    if not hasattr(neuron.h, templatename):
        neuron.h.load_file(1, "template.hoc")

    return templatename

def assess_f_I(cell_name):
    neuron.h.load_file('stdrun.hoc')
    neuron.h.load_file('import3d.hoc')

    #Loading the neuron models of interest

    load_mechanisms_from_neuron_model(cell_name)
    load_cell_properties(cell_name)

def generate_LFP_for_single_neuron(electrode, cell_name, params):
    cwd = os.getcwd()
    neuron.h.load_file('stdrun.hoc')
    neuron.h.load_file('import3d.hoc')

    #Loading the neuron models of interest
    neuron_model = 'home/baran/Desktop/neuron_models/'+cell_name


    load_mechanisms_from_neuron_model(cell_name)
    templatename = load_cell_properties(cell_name)
    os.chdir(cwd + neuron_model)
    morphologyfile = glob(os.path.join('morphology', '*'))[0]
    add_synapses = False

    cell = LFPy.TemplateCell(morphology=morphologyfile,
                        templatefile=(os.path.join(neuron_model, 'template.hoc')),
                        templatename=templatename,
                        templateargs=1 if add_synapses else 0,
                        tstop=params['tstop'],
                        dt=params['dt'],
                        nsegs_method=None,pt3d=True,verbose=True)

    #Add stimulation electrode
    pointProcess = LFPy.StimIntElectrode(cell, **params['PointProcParams'])

    cell.simulate(electrode=electrode)
    os.chdir(cwd)
    return electrode, cell
