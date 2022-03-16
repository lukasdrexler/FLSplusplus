import matplotlib.pyplot as plt
from sklearn import cluster
import argparse
from PIL import Image
import numpy as np




def _build_arg_parser():

    arg_parser = argparse.ArgumentParser(description=""
                                                     "Argparser for this program")

    arg_parser.add_argument(
        "-f", "--file",
        type=argparse.FileType('r'),
        required=True,
        help="Picture (as txt-file with rgb-values) to be clustered/compressed"
    )


    return arg_parser

if __name__ == '__main__':

    args = _build_arg_parser().parse_args()

    #X_from_txt = np.flip(np.genfromtxt('datasets_large/frymire.txt',dtype=int),0)
    img_from_jpg = np.asarray(Image.open('datasets_large/chamchaude-g6903e2e5a_1920.jpg'))

    img_dims = img_from_jpg.shape

    #img_from_txt = np.flip(X_from_txt.reshape(img_dims[0], img_dims[1], img_dims[2]),0)

    X_from_jpg = img_from_jpg.reshape(img_dims[0] * img_dims[1], img_dims[2])

    while True:

        kmeans = cluster.KMeans(init='k-means++',
                                    n_clusters=4,
                                    n_init=1
                                )

        alspp = cluster.KMeans(init='k-means++',
                                n_clusters=4,
                                n_init=1,
                                algorithm='als++',
                                depth=3,
                                search_steps=1,
                                norm_it=7
                                )

        kmeans.fit(X_from_jpg)
        alspp.fit_new(X_from_jpg)

        if alspp.inertia_ / kmeans.inertia_ < 0.9:

            print(kmeans.inertia_)
            print(alspp.inertia_)
            print(alspp.inertia_/kmeans.inertia_)

            kmeans_compressed = kmeans.cluster_centers_[kmeans.labels_]
            kmeans_compressed = np.clip(kmeans_compressed.round().astype('uint8'), 0, 255)
            kmeans_compressed = kmeans_compressed.reshape(img_dims[0], img_dims[1], img_dims[2])

            alspp_compressed = alspp.cluster_centers_[alspp.labels_]
            alspp_compressed = np.clip(alspp_compressed.round().astype('uint8'), 0, 255)
            alspp_compressed = alspp_compressed.reshape(img_dims[0], img_dims[1], img_dims[2])

            fig, ax = plt.subplots(1, 3, figsize=(16, 8))
            ax[0].imshow(img_from_jpg)
            ax[0].set_title('Img from jpg')
            ax[1].imshow(kmeans_compressed)
            ax[1].set_title('Compressed by kmeans')
            ax[2].imshow(alspp_compressed)
            ax[2].set_title('Compressed by als++')
            ax[0].axis('off')
            ax[1].axis('off')
            ax[2].axis('off')
            plt.show()



            fig1, ax1 = plt.subplots(1, 1)
            fig2, ax2 = plt.subplots(1, 1)
            fig3, ax3 = plt.subplots(1, 1)

            ax1.imshow(img_from_jpg)
            ax2.imshow(kmeans_compressed)
            ax3.imshow(alspp_compressed)

            ax1.axis('off')
            ax2.axis('off')
            ax3.axis('off')

            fig1.savefig("original.svg", format='svg', dpi=1200)
            fig2.savefig("kmeans.svg", format='svg', dpi=1200)
            fig3.savefig("alspp.svg", format='svg', dpi=1200)

            break