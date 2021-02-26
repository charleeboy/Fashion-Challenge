from visualizer import Visualizer

vis = Visualizer(epochs=10)

for i in range(1, 10):
    for j in range(15,30):
        vis.add_loss(i, j)
    vis.add_trainAccurancy(i, i/100)
    vis.add_validationAccurancyLoss(i, i/103, i/108 )

print(vis.get_epochs())
vis.get_basicLossAccurancyPlot()

