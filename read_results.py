import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

eval_loss_1 = np.array(pd.read_pickle(r'loss_record/eval_loss_DLO1.pkl'))
eval_step_1 = np.array(pd.read_pickle(r'loss_record/eval_epoch_DLO1.pkl'))
# eval_loss_1_twisting = np.array(pd.read_pickle(r'loss_record/twisting_eval_loss_DLO1.pkl'))
# eval_step_1_twisting = np.array(pd.read_pickle(r'loss_record/twisting_eval_epoch_DLO1.pkl')) + 540

eval_loss_2 = np.array(pd.read_pickle(r'loss_record/eval_loss_DLO2.pkl'))
eval_step_2 = np.array(pd.read_pickle(r'loss_record/eval_epoch_DLO2.pkl'))
# # eval_loss_2_twisting = np.array(pd.read_pickle(r'loss_record/twisting_eval_epoch_DLO2.pkl'))
# # eval_step_2_twisting = np.array(pd.read_pickle(r'loss_record/twisting_eval_loss_DLO2.pkl'))
#
eval_loss_3 = np.array(pd.read_pickle(r'loss_record/eval_loss_DLO3.pkl'))
eval_step_3 = np.array(pd.read_pickle(r'loss_record/eval_epoch_DLO3.pkl'))
# # eval_step_3_twisting = np.array(pd.read_pickle(r'loss_record/twisting_eval_epoch_DLO3.pkl'))
# # eval_loss_3_twisting = np.array(pd.read_pickle(r'loss_record/twisting_eval_loss_DLO3.pkl'))
#
eval_loss_4 = np.array(pd.read_pickle(r'loss_record/eval_loss_DLO4.pkl'))
eval_step_4 = np.array(pd.read_pickle(r'loss_record/eval_epoch_DLO4.pkl'))
#
eval_loss_5 = np.array(pd.read_pickle(r'loss_record/eval_loss_DLO5.pkl'))
eval_step_5 = np.array(pd.read_pickle(r'loss_record/eval_epoch_DLO5.pkl'))
# print(eval_loss_1)
# print(eval_loss_2)
# print(eval_loss_3)
# print(eval_loss_4)
# print(eval_loss_5)

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5)
fig.set_figheight(10)
fig.set_figwidth(30)

line1 = ax1.plot(eval_step_1, eval_loss_1, label='DLO1')
line2 = ax2.plot(eval_step_2, eval_loss_2, label='DLO2')
line3 = ax3.plot(eval_step_3, eval_loss_3, label='DLO3')
line4 = ax4.plot(eval_step_4, eval_loss_4, label='DLO4')
line5 = ax5.plot(eval_step_5, eval_loss_5, label='DLO5')

# # # #
ax1.set_title('Eval: DLO1')
ax1.set_xlabel('Training Iterations')

ax2.set_title('Eval: DLO2')
ax2.set_xlabel('Training Iterations')

ax3.set_title('Eval: DLO3')
ax3.set_xlabel('Training Iterations')

ax4.set_title('Eval: DLO4')
ax4.set_xlabel('Training Iterations')

ax1.grid(which = "minor")
ax1.minorticks_on()
ax2.grid(which = "minor")
ax2.minorticks_on()
ax3.grid(which = "minor")
ax3.minorticks_on()
ax4.grid(which = "minor")
ax4.minorticks_on()
ax5.grid(which = "minor")
ax5.minorticks_on()
plt.legend()
plt.show()


