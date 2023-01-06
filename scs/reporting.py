import numpy as np

import sys
sys.path.append('/zfsauton2/home/vsanil/projects/aihc/cardiovascular_clinical_trials/sparse_cox_subgroup')

from scs.experiments import get_predictions

from auton_survival.reporting import plot_kaplanmeier
from auton_survival.metrics import treatment_effect

def plot_survival_curves(model, rank, d, features, outcomes, interventions, intervention,
												 selected_features, ylim=(0.5, 1.0), xlim=(0, 6*365.25), title="Top ",
												 save=False):

	preds = get_predictions(model, features, d, selected_features=selected_features)

	intrank = rank 

	rank = int(rank*len(outcomes)/100)

	prioritization_index = np.argsort(preds)
	predictions = np.zeros_like(prioritization_index)
	predictions[prioritization_index[-rank:]] = 1

	# hr = treatment_effect('hazard_ratio', outcomes[predictions==0], 
	# 											(interventions==input['intervention'])[predictions==0])

	# from matplotlib import pyplot as plt
	# plot_kaplanmeier(outcomes[predictions==0],
	# 								(interventions[predictions==0]))
	# plt.title('n:'+str(sum(predictions==0)) + " " + "Hr:"+str(hr))
	# plt.ylim(0.5, 1.0)
	# plt.show()

	hr = treatment_effect('hazard_ratio', outcomes[predictions==1], 
												(interventions==intervention)[predictions==1])

	from matplotlib import pyplot as plt

	fig = plt.figure(figsize=(8, 6))

	plot_kaplanmeier(outcomes[predictions==1],
									(interventions[predictions==1]))

	years = [0, 1, 2, 3, 4, 5, 6]

	
	plt.grid(ls=':')
	plt.title("Top " + str(intrank)+ "\% " + title, fontsize=21)

	plt.legend(fontsize=21, shadow=True)
	plt.yticks(fontsize=18)
	plt.xticks([365.25*year for year in years], ['0']+[str(year)+ " " + "Years" for year in years[1:]],
						 fontsize=18)
	plt.ylim(ylim)
	plt.xlim(xlim)
	plt.xlabel(r"\textbf{Time in Years} $\rightarrow$", fontsize=21)
	plt.ylabel(r"\textbf{Event-free Survival} $\rightarrow$", fontsize=21)
	
	plt.text(0.1, 0.1, s='Size: '+str(sum(predictions==1)) + ", " + "HR: "+str(round(hr, 2)),
					 horizontalalignment='left', verticalalignment='baseline', 
					 transform=fig.gca().transAxes,
					 fontsize=21)

	if save:
		plt.savefig('Surv_curves.pdf', format='pdf', bbox_inches='tight')
	
	plt.show()

def plot_all_results(input, results, outcomes, interventions, selected_features):

	from matplotlib import pyplot as plt
	plt.rcParams['font.family'] = 'serif'
	plt.rcParams['text.usetex'] = True

	titles = ['Diminished Treatment Effect', 'Enhanced Treatment Effect']

	for result in results:

		if result=='RMST': ate = treatment_effect('restricted_mean', outcomes, interventions==input['intervention'], horizons=5*365.25)
		elif result=='HR': ate = treatment_effect('hazard_ratio', outcomes, interventions==input['intervention'])
		elif result=='RISK': ate = treatment_effect('survival_at', outcomes, interventions==input['intervention'], horizons=5*365.25)

		for d in results[result]:

			# cmap = plt.get_cmap('cividis')
			# colors = cmap(np.linspace(0,1,8)) #get 10 colors along the full range of hsv colormap

			fig, ax = plt.subplots(figsize=(10, 5))

			# ax.set_prop_cycle(color=colors) #set our 10 colors to the property cycle


			for i, model in enumerate(results[result][d]):
				# plt.plot(ranks, results[result][d][model].mean(axis=1), label=model, ls='--', lw=1, markerfacecolor='white', marker='s', markersize=10)

				plt.errorbar(x=np.array(input['ranks'])+i*1.25 -2.5, y=results[result][d][model].mean(axis=1), color='C'+str(i),
											yerr=results[result][d][model].std(axis=1),	marker='o', markersize=7.5, markerfacecolor='C'+str(i),
											label=r'\texttt{'+model+'}', ls='none', lw=2,elinewidth=2.5, capthick=2.5, capsize=5)

			
				# plt.errorbar(x=ranks, y=np.array(rmsts[d][model])[: ,:, 0].mean(axis=1), 
				# 				yerr = np.array(rmsts[d][model])[: ,:, 0].std(axis=1), 
				# 				label=model, lw=1, capsize=5)

			plt.title(r'\textbf{'+titles[d]+'}', fontsize=21)

			plt.plot(range(0, 105), [ate]*len(range(0, 105)), ls='--', color='k', label='ATE', marker='x', lw=1, markevery=5, markersize=10, zorder=-200, markerfacecolor='white')

			#plt.grid(ls=':')
			plt.legend(fontsize=21, shadow=True, ncol=2, handletextpad=0, columnspacing=0.5, loc=1) 
			plt.xlim(input['ranks'][0]-5, input['ranks'][-1]+5)

			plt.xticks(input['ranks'], [str(xt)+'\%' for xt in input['ranks']], fontsize=21)
			plt.yticks(fontsize=18)
			plt.ylabel(r"\textbf{CATE (" +result+")}", fontsize=21)
			plt.xlabel(r"\textbf{Subgroup Size} $\longrightarrow$", fontsize=21)

			plt.tight_layout()
			plt.savefig("results/"+input['dataset']+'_'+str(selected_features)+'_'+result+'_'+str(d)+'.pdf')
			plt.show()