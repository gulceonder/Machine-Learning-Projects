# part of the code from: https://pythonhosted.org/scikit-fuzzy/auto_examples/plot_tipping_problem_newapi.html
import os

from matplotlib import pyplot as plt
from matplotlib import ticker
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


# class object to simulate vaccinating people
class Vaccination(object):

    def __init__(self):

        self.vaccinated_percentage_curve_ = [0.] # percentage of the vaccinated people
        self.vaccination_rate_curve_ = [0.]   # percentage / day
        self.vaccination_control_curve_ = [0.]    # percentage / day

    def _vaccinationFailureRate(self):

        p = self.vaccinated_percentage_curve_[-1]

        return .5 * p * p

    # method to apply the control
    def vaccinatePeople(self, vaccination_control):
        """
    applies the control signal to vaccinate people and updates the status curves

    Arguments:
    ----------

    vaccination_control: float, vaccination rate to be added to the current vaccination rate

        """

        # update vaccination rate according to control signal
        curr_vaccination_rate = self.vaccination_rate_curve_[-1]
        vaccination_rate = max(0., min(.6, curr_vaccination_rate + vaccination_control))

        effective_vaccination_rate = vaccination_rate - self._vaccinationFailureRate()

        # update the vaccinated percentage after 0.1 Day of vaccination with the current rate
        vaccination_percentage = \
            min(1., self.vaccinated_percentage_curve_[-1] + effective_vaccination_rate * .1)

        # update status curves
        self.vaccinated_percentage_curve_.append(vaccination_percentage)
        self.vaccination_rate_curve_.append(vaccination_rate)
        self.vaccination_control_curve_.append(vaccination_control)

    # method to obtain measurements
    def checkVaccinationStatus(self):
        """
    returns the current vaccinated percentage and vaccination rate as a two-tuple
    (vaccinated_percentage, vaccination_rate)

    Returns:
    ----------

    (vaccinated_percentage, effective_vaccination_rate): (float, float) tuple,
                    vaccination percentage and rate to be used by the controller

        """

        vaccinated_percentage = self.vaccinated_percentage_curve_[-1]
        effective_vaccination_rate = \
            self.vaccination_rate_curve_[-1] - self._vaccinationFailureRate()

        return (vaccinated_percentage, effective_vaccination_rate)

    # method to visualize the results for the homework
    def viewVaccination(self, point_ss, vaccination_cost, save_dir='', filename='vaccination', show_plot=True):
        """
        plots multiple curves for the vaccination and
            saves the resultant plot as a png image

        Arguments:
        ----------

        point_ss: int, the estimated iteration index at which the system is at steady state

        vaccination_cost: float, the estimated cost of the vaccination until the steady state

        save_dir: string, indicating the path to directory where the plot image is to be saved

        filename: string, indicating the name of the image file. Note that .png will be automatically
        appended to the filename.

        show_plot: bool, whether the figure is to be shown

        Example:
        --------

        visualizing the results of the vaccination

        # assume many control signals have been consecutively applied to vaccine people

        >>> my_vaccine = Vaccination()

        >>> my_vaccine.vaccinatePeople(vaccination_control) # assume this has been repeated many times

        >>> # assume state state index and the vaccination cost have been computed

        >>> # as point_ss=k and vaccination_cost=c

        >>> my_vaccine.viewVaccination(point_ss=k, vaccination_cost=c,
        >>>                      save_dir='some\location\to\save', filename='vaccination')

        """

        color_list = ['#ff0000', '#32CD32', '#0000ff', '#d2691e', '#ff00ff', '#000000', '#373788']
        style_list = ['-', '--']

        num_plots = 3

        plot_curve_args = [{'c': color_list[k],
                            'linestyle': style_list[0],
                            'linewidth': 3} for k in range(num_plots)]

        plot_vert_args = [{'c': color_list[k],
                            'linestyle': style_list[1],
                            'linewidth': 3} for k in range(num_plots)]

        font_size = 18

        fig, axes = plt.subplots(3, 1, figsize=(16, 12))

        day_x = [i * .1 for i in range(len(self.vaccinated_percentage_curve_))]
        x_ticks = day_x[::10]

        # vaccinated population
        ax = axes[0]
        ax.set_title('vaccinated population percentage over days', loc='left', fontsize=font_size)
        ax.plot(day_x[:point_ss+1], self.vaccinated_percentage_curve_[:point_ss+1], **plot_curve_args[0])
        ax.plot(day_x[point_ss:], self.vaccinated_percentage_curve_[point_ss:], **plot_curve_args[1])
        ax.plot([day_x[point_ss]] * 2, [0, self.vaccinated_percentage_curve_[point_ss]], **plot_vert_args[2])


        ax.set_xlabel(xlabel='day', fontsize=font_size)
        ax.set_ylabel(ylabel='vaccinated population %', fontsize=font_size)
        ax.set_xticks(x_ticks)
        ax.xaxis.set_minor_locator(ticker.FixedLocator([day_x[point_ss]]))
        ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
        ax.tick_params(which='minor', length=17, color='b', labelsize=13)
        ax.tick_params(labelsize=15)
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0)
        ax.grid(True, lw = 1, ls = '--', c = '.75')

        # vaccination rate
        ax = axes[1]
        ax.set_title('vaccination rate over days', loc='left', fontsize=font_size)
        ax.plot(day_x[:point_ss + 1], self.vaccination_rate_curve_[:point_ss + 1],
                **plot_curve_args[0])
        ax.plot(day_x[point_ss:], self.vaccination_rate_curve_[point_ss:],
                **plot_curve_args[1])
        ax.plot([day_x[point_ss]] * 2, [0, self.vaccination_rate_curve_[point_ss]],
                **plot_vert_args[2])
        ax.fill_between(day_x[:point_ss + 1], 0, self.vaccination_rate_curve_[:point_ss + 1],
                        facecolor='#FF69B4', alpha=0.7)

        ax.text(1.5, .01, 'cost = %.2f'%vaccination_cost,
                horizontalalignment='center', fontsize=font_size)

        ax.set_xlabel(xlabel='day', fontsize=font_size)
        ax.set_ylabel(ylabel='vaccination rate (%/day)', fontsize=font_size)
        ax.set_xticks(x_ticks)
        ax.xaxis.set_minor_locator(ticker.FixedLocator([day_x[point_ss]]))
        ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
        ax.tick_params(which='minor', length=17, color='b', labelsize=13)
        ax.tick_params(labelsize=15)
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0)
        ax.grid(True, lw=1, ls='--', c='.75')

        # vaccination rate control
        ax = axes[2]
        ax.set_title('vaccination rate control over days', loc='left', fontsize=font_size)
        ax.plot(day_x[:point_ss + 1], self.vaccination_control_curve_[:point_ss + 1],
                **plot_curve_args[0])
        ax.plot(day_x[point_ss:], self.vaccination_control_curve_[point_ss:],
                **plot_curve_args[1])
        y_min = ax.get_ylim()[0]
        ax.plot([day_x[point_ss]] * 2,
                [y_min, self.vaccination_control_curve_[point_ss]],
                **plot_vert_args[2])

        ax.set_xlabel(xlabel='day', fontsize=font_size)
        ax.set_ylabel(ylabel='vaccination rate control (%/day)', fontsize=font_size)
        ax.set_xticks(x_ticks)
        ax.xaxis.set_minor_locator(ticker.FixedLocator([day_x[point_ss]]))
        ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
        ax.tick_params(which='minor', length=17, color='b', labelsize=13)
        ax.tick_params(labelsize=15)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=y_min)
        ax.grid(True, lw=1, ls='--', c='.75')

        if show_plot:
            plt.show()

        fig.savefig(os.path.join(save_dir, filename + '.png'))


#create vaccination class object and check current status to start vaccination process
vaccination=Vaccination()
(pi,pi_dot)= vaccination.checkVaccinationStatus()

# New Antecedent/Consequent objects hold universe variables and membership
# functions
#antecedent for input consequent for output
cur_vac = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'Currently Vaccinated')

out_rate = ctrl.Consequent(np.arange(-0.2, 0.25, 0.05), 'Output Rate')

# Custom membership functions can be built interactively with a familiar,
# Pythonic API
#trapmf for trapezoid functions 4 edges provide
#trimf for triangular functions 3 edges provided

cur_vac['low'] = fuzz.trapmf(cur_vac.universe, [0, 0, 0.3, 0.6])
cur_vac['medium'] = fuzz.trimf(cur_vac.universe, [0.3, 0.6, 0.9])
cur_vac['high'] = fuzz.trapmf(cur_vac.universe, [0.6, 0.8, 1.0, 1.0])

out_rate['low'] = fuzz.trapmf(out_rate.universe, [-0.2,-0.2, -0.15, 0])
out_rate['medium'] = fuzz.trimf(out_rate.universe, [-0.1, 0, 0.1])
out_rate['high'] = fuzz.trapmf(out_rate.universe, [0, 0.15, 0.20,0.20])


# View membership functions
cur_vac.view()
out_rate.view()

#set rules
rule1 = ctrl.Rule(cur_vac['low'] , out_rate['high'])
rule2 = ctrl.Rule(cur_vac['medium'], out_rate['medium'])
rule3 = ctrl.Rule(cur_vac['high'] , out_rate['low'])

#rule1.view()

#create a control system to simulate  
vaccination_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])

vaccinate = ctrl.ControlSystemSimulation(vaccination_ctrl)
# rate=0

# effective= [] 
# effective.append(pi_dot) 

# Pass inputs to the ControlSystem using Antecedent labels with Pythonic API
# Note: if you like passing many inputs all at once, use .inputs(dict_of_data)
for i in range(200):
    
    # print(vaccination.vaccination_rate_curve_)
    vaccinate.input['Currently Vaccinated'] = pi


    # Crunch the numbers
    vaccinate.compute()

    #viusalize output
    #print (vaccinate.output['Output Rate'])
    #out_rate.view(sim=vaccinate)

    vaccination.vaccinatePeople(vaccinate.output['Output Rate'])


    (pi,pi_dot)= vaccination.checkVaccinationStatus()
    #effective.append(pi_dot)




#plot to see vaccinated percentage curve
plt.figure()
x=np.arange(0,201)
plt.plot(x,vaccination.vaccinated_percentage_curve_)
plt.xlim([0, 200])
plt.title( "Vaccinated Percentage Curve")
plt.show()

#plot to see Effective Vaccination Rate Curve
# plt.plot(x,effective)
# plt.title("Effective Vaccination Rate Curve")
# plt.show()

#plot to see Vaccination Rate Curve
x=np.arange(0,201)
plt.plot(x,vaccination.vaccination_rate_curve_)
plt.xlim([0, 200])
plt.title( "Vaccination Rate Curve")
plt.show()

#compute cost from vaccination rate curve
vac_cost=vaccination.vaccination_rate_curve_[0:125]
cost=np.sum(vac_cost) *100
# print(cost*100)

#view vaccination process
vaccination.viewVaccination(125, cost, save_dir='/Users/gulceonder/Desktop/webpage/webdevelopment', filename='vsc', show_plot=True)




