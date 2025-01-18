# Recommender System

> Реализирайте система за генериране на препоръка за закупуване на книги
> (recommender system). Може да използвате мярка за сходство между потребителите
> на книги въз основа на техните рейтингови записи или друго сходство, за да
> отправяте препоръки към читателите.

[DATASET](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset)

За информация по [темата](https://towardsdatascience.com/how-did-we-build-bookrecommender-systems-in-an-hour-the-fundamentals-dfee054f978e)
и [темата](https://towardsdatascience.com/how-did-we-build-book-recommender-systems-in-anhour-%20part-2-k-nearest-neighbors-and-matrix-c04b3c2ef55c)

## Starter Kernel(s)

[Recom I: Data Understanding and Simple Recommendation](https://www.kaggle.com/arashnic/recom-i-data-understanding-and-simple-recomm)

## Повече информация

### My Recommendation Article Series in Medium:

[Evolution of Recommendation Algorithms, Part I: Fundamentals , History Overview, Core and Classical Algorithms](https://medium.com/@anicomanesh/evolution-of-recommendation-algorithms-part-i-fundamentals-and-classical-recommendation-bb1c0bce78a9)

## Препоръки на базата на повече от 2 параметъра чрез регресия, грешна класификация и грануларни изчисления:
> например: **рейтинг** - базов компонент на препоръката, **възрастово групиране на читателите** и **жанрово разпределение на книгите**;
- **Регресия**: https://www.sciencedirect.com/science/article/abs/pii/S0020025516301669
- **Грешна класификация**: https://www.sciencedirect.com/topics/engineering/misclassification
- **Грануларни изчисления**: https://www.sciencedirect.com/topics/computer-science/granular-computing

## Кратко описание на задачата:
> Система за препоръки на книги, като препоръките се базират на рейтинг от читателите/потребителите чрез корелация по рейтинг и прилагането на k-NN алгоритъма. На данните би могло да се приложи ре-филтрация по жанрове и по възраст на читателите (евентуално). Основен проблем, който би следвало да се реши преди ре-филтрацията е, че 40% от читателите не са посочили възрастта си, което води до нужда от поправки/зачистване в данните преди ре-филтрацията.
