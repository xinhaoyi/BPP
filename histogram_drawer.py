from pyecharts import options as opts
from pyecharts.charts import Bar


class Drawer:

    def __init__(self, num_of_hyper_edges, num_of_nodes, dimensionality, pathway_name, path):
        self.num_of_hyper_edges = num_of_hyper_edges
        self.num_of_nodes = num_of_nodes
        self.dimensionality = dimensionality
        self.pathway_name = pathway_name
        self.path = path

    def generate_histogram(self):
        x1 = [self.pathway_name]
        y1 = [self.num_of_hyper_edges]
        y2 = [self.num_of_nodes]
        y3 = [self.dimensionality]
        name_of_file = self.pathway_name + ".html"
        path = self.path + '/' + name_of_file
        bar = (
            Bar()
            .add_xaxis(x1)
            .add_yaxis("Hyper Edges(Reactions)", y1)
            .add_yaxis("Nodes(Physical Entity)", y2)
            .add_yaxis("Dimensionality", y3)
            .set_global_opts(title_opts=opts.TitleOpts(title=self.pathway_name)))

        bar.render(path)
