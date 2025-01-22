# Recommender System

> Реализирайте система за генериране на препоръка за закупуване на книги
> (recommender system). Може да използвате мярка за сходство между потребителите
> на книги въз основа на техните рейтингови записи или друго сходство, за да
> отправяте препоръки към читателите.

## Кратко описание на задачата

> Система за препоръки на книги, като препоръките се базират на рейтинг от читателите/потребителите. На данните би могло да се приложи ре-филтрация по възраст на читателите. Основен проблем, който би следвало да се реши преди ре-филтрацията е, че около 40% от читателите не са посочили възрастта си, което води до нужда от поправки/зачистване в данните преди ре-филтрацията, което на по-късен етап установихме, че води до загуба на информация и неточности в модела. Естествено, в следната документация предлагаме пълния процес, през който преминахме в изграждането на система за препоръки на книги. 
>
> Забележка: Преди да достигнем крайната реализация на задачата си преминахме през няколко различни потенциални подхода за решаване на подобна задача, но основен подход, на който се спряхме, за реализация на задачата ни е k-NN алгоритъма във варианта му на a-NN (k-NN алгоритъм с приближение). За да изведем съображенията, които ни насочиха да тръгнем в тази посока би било редно да опишем подходите и експериментите, които проведохме, заедно с техните резултати и изводи, както и да покажем с какво a-NN подхода превъзхожда всички останали, изпробвани алгоритми в условностите на задачата ни. Повече информация за процеса на имплементация, както и за преценката за нужния ни алгоритъм може да намерите в графа Използвани алгоритми на документацията.
>
> Забележка: Кодът е разработен на Python със съответно необходимите му библиотеки като:
> 1. Numpy
> 2. pandas
> 3. sklearn.neighbors
> 4. scikit-learn (от sklearn.neighbors)
> 5. sklearn.metrics
> 6. sklearn.model_selection
> 7. scipy.stats
> 8. scipy.sparse 
> 9. warnings
> 
> По-подробно описание на всяка една от тези библиотеки може да открита в графа **Формулировка на задачата** на документацията.

## Данните

> Таблиците, които описваме по-долу са взети от [DATASET](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset). Ресурсът е цитиран и по-долу в графа ресурси.

### Users

> Таблица, която съдържа информация за потребителите, наричани още читатели в рамките на нашата система. Информацията, която се съхранява за читателите е техните уникални идентификационни номера (User-ID), локациите им (Location) и възрастта им. Потребителските идентификатори са анонимизирани и съпоставими с цели числа. Данните за местоположението на читателите и тяхната възраст се предоставят, ако има такива. В противен случай тези полета приемат NULL стойности. Това предполага, че ако те бъдат включвани в методите за категоризация и препоръки на книги на други потребители, трябва да се подсигурим за коректността на данните и евентуално да ги почистим от NULL стойности, тъй като те не внасят никаква информация и пречат на процеса по препоръка на книги.

### Books

> Таблица, която предоставя информация за книгите, от които имаме в наличност и ще можем да препоръчваме на нашите читатели. Книгите са уникално идентифицируеми от техните ISBN номера като трябва предварително да се погрижим, че всички ISBN номера са уникални и валидни. Таблицата също така предоставя информация за съдържанието на книгата като заглавие на книгата (Book-Title), автор на книгата (Book-Author), година на публикация на книгата (Year-Of-Publication) и издателство на книгата (Publisher). **Тази информация е предоставена от Amazon Web Services.** Трябва да се отбележи също така, че при наличие на съавторство над някоя книга, само имената на първият автор са предоставени. В таблицата също така се съхранява информация за линка към корицата на съответната книга (линк към уеб сайта на Amazon). Тези линкове може да се показват под различни форми (Image-URL-S - къси, Image-URL-M - средно дълги, Image-URL-L - дълги).

### Ratings

> Таблица, която описва оценките на книгите. В нея се съдържа информация кой потребител (User-ID) коя книга (ISBN) а е оценил и каква е била самата оценка (число в интервала от 1 до 10, вписана в полето Book-Rating). Ако някоя книга все още не е била оценена от даден потребител, оценката, която автоматично се вписва в полето за оценка (Book-Rating) е 0.

## Организация на файловете
> По-надолу описваме структурата на проекта си - неговите папки, както и съдържанието им, за прегледност и по-голямо разбиране при работа с кода на нашата система.

- docs - папка, която съдържа документацията на проекта (, тоест текущо разглеждания от Вас файл), както и папка resources и прилежащите й файлове, необходими за изграждането на тази документация.
	- resources - папка, съдържаща всички необходими ресурсни файлове за създаването на тази документация (например скрийншотите, включени в нея).
- public- папка, съдържаща глобално достъпни (и от двамата разработчици) файлове, необходими за осмислянето на задачата и построението на решението, такова, каквото е изложено в този документ. 
- src - папка, съдържаща данните за обработка, изходния код на системата и Jupiter notebook-а, описващ достатъчно подробно кода и прилежащите му примери. Цялото това съдържание условно се разделя в следните три папки:
	- code - папка, която съдържа изходния код на всички проведени експерименти, описани в този документ, както и изходния код на крайния алгоритъм, използван за решаването на проблема.
	- data - папка, съдържаща всички CSV таблици, използвани в процеса на разработка на системата.
	- jupyter - папка, съдържаща ipynb файла на въпросния Jupiter notebook, използван за демонстрация на работата на системата и документация на самата демонстрация. 

## Описание на функционалността

> Препоръки за книги могат да се правят на базата на техния рейтинг, на базата на възрастовите групи сред потребителите на системата ни, които се очертават или на база двата компонента едновременно. Ако обаче препоръчваме книги на база двата компонента би следвало да се провери първо дали тези два компонента зависят един от друг по някакъв начин иначе една такава препоръка не би била състоятелна. Все пак ако се окаже, че има смисъл от нея трябва да имаме предвид подходите за реализацията на система с подобен тип препоръки.

**Следователно разглеждаме следните варианти за препоръки на базата на повече от 2 параметъра:**

### Препоръки на базата на повече от 2 параметъра чрез регресия и гранулирани изчисления:

> например: **рейтинг** - базов компонент на препоръката, **възрастово групиране на читателите** и **жанрово разпределение на книгите**;

- **Регресия**: [Regression article](https://www.sciencedirect.com/science/article/abs/pii/S0020025516301669)
- **Гранулирани изчисления**: [Granular computing article](https://www.sciencedirect.com/topics/computer-science/granular-computing)

> Очертават се основно 3 подхода:
> - **линейна регресия с параметри рейтинг, възрасти, жанрове** (като ги свеждаме до категорийни или числови променливи) или **друг вид регресия при по-сложни модели**;
> - **гранулиране на данните** (групираме жанровете в нови смесени жанрове и/или групираме възрастите в няколко възрастови групи, за да опростим модела и идентифицираме патърните и трендовете във всеки един от клъстърите);
> - **смесен подход** - клъстеризация на данните на база възраст и какви жанрове книги предпочитат за опростяване + регресия за всеки клъстер за по-прецизни изчисления за конкретната група хора, което да води до по-голяма персонализация на резултатите;

#### Идеи за реализация 
- k-NN алгоритъм
	1. Зареждаме необходимите библиотеки.
	2. Прочитаме данните в dataframe-ове и изследваме данните (големина, допълнителна информация за dataframe-овете, сортиране на данните, отпечатване на данните, зачистване от незначещи записи и 	т.н.).
 	3. Създаваме k-NN модела като използваме Евклидова метрика, Манхатанските разстояния като метрика или друга, подходяща за случая метрика. 
	4. Трениране на модела по някой от следните методи:
		- train/test - използва разделение на данните на тренировъчно и тестово множество, при което данните се разделят на тренировъчно и отделно тестово множество. Този подход позволява проста оценка на производителността на модела, но не използва пълноценно всички налични данни за обучение.
		- k-fold - използва k-кратна кръстосана валидация, при която множеството данни се разделя на k равни части. Моделът се обучава и оценява многократно, което осигурява по-цялостна оценка на производителността му и намалява влиянието на вариациите в данните. Въпреки това, този подход изисква много повече изчислителна мощ.
		- build full trainset - използва метода build_full_trainset(), което създава тренировъчно множество, включващо всички налични оценки от множеството данни. Тази опция е полезна, когато множеството данни е сравнително малко и е желателно максимално ефективно използване на информацията за обучение. Това позволява на модела да се учи от целия набор данни, което потенциално подобрява точността.
  	5. Предсказваме най-популярните книги и препоръчваме на читателите някои от тях.
  
[KNN_Regression_approach](https://medium.com/@leam.a.murphy/personalized-book-recommendations-with-k-nearest-neighbors-442ce4dad44c)

[GeekforGeeks-Recommender Systems using KNN](https://www.geeksforgeeks.org/recommender-systems-using-knn/)

- k-NN алгоритъм с линейна регресия за оценката влиянието на годините на читателите върху рейтинга на книгите и грануларни изчисления на разстоянията между текущо изследваната книга и нейните k на брой най-близки съседи чрез определена метрика (Евклидова, Манхатански разстояния, cousine и други) - стъпките са сходни като тези на горния алгоритъм, но с включване на корелация между рейтингите и възрастите на читателите, регресия при положителна оценка за линейна връзка между двата компонента и грануларни изчисления, базирани на следната формула:

$$
d_w(X, Y) = \sqrt{\sum_{i=1}^{n} w_i \cdot (x_i - y_i)^2}
$$

където:
- \( n \) - броят компоненти, участващи в препоръката;
- \( w_i \) - теглото (приоритетът), с който участва \( i \)-ят компонент в препоръката;
- \( x_i \) и \( y_i \) - са координатите на \( i \)-ия компонент в препоръката.

> Тези метрики обаче не са достатъчно добри за нашата задача, защото те допускат дори и вземането на всички примери (в случая книги), които са на горе-долу еднакво разстояние от текущо тествания, а **cousine метриката** - взема предвид и ъгъла, под който се намира съседа на текущо тествания пример спрямо текущо тествания пример и въз основа на това колко голям е ъгълът решава кой съсед (в случая книга) да добави в препоръчаните.

## Експерименти
### k-NN алгоритъм
> k-NN алгоритъма е класификационен алгоритъм, който се базира на изчисления на разстоянията между текущо изследвания тестови пример и неговите k на брой най-близки съседи, така че да се открият най-добрите примери от някаква извадка данни и според тази класификация, те да бъдат препоръчани на потребителя на системата (в случая говорим за книги, които трябва да се препоръчат на определена група читатели със сходства помежду си). За тази цел ни трябва определена метрика, която често пъти е Евклидово или Мнахатанско разстояние между дадените тестови примери и техните най-близки съседи (за простота).
> - Евклидово разстояние:
>   Ако точката \( A(x_1, y_1) \) и точката \( B(x_2, y_2) \), то

$$
d_{AB} = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2}
$$

[Euclidean distance](https://www.geeksforgeeks.org/euclidean-distance)
> - Манхатанско разстояние:
>   Ако точката \( A(x_1, y_1) \) и точката \( B(x_2, y_2) \), то

$$
d_{AB} = |x_1 - x_2| + |y_1 - y_2|
$$

[Manhattan distance](https://www.datacamp.com/tutorial/manhattan-distance)

### k-NN алгоритъма в съчетание с регресия и грануларни изчисления по Евклид на разстоянията между изследвания пример и неговите k най-близки съседи (в случая книги) за отправяне на препоръки към потребителите на системата (в случая читателите)
- **Скрийншот на препоръчаните книги за 2-ма читатели с изцяло различни характеристики**:

![recommendations_knn_reg_granular_compute](https://github.com/user-attachments/assets/5b822f7d-2c10-499d-8ae5-e22b85d5d962)
  
- **експериментален алгоритъм в стъпки**:
	1. Включване на необходимите библиотеки и пакети.
	2. Зареждане на данните.
	3. Обединяване на данните за рейтингите на книгите (всички данни от таблицата с рейтингите) с годините на читателите.
	4. Почистване на данните от невалидни такива (NaN стойности).
	5. Подравняване на размерите на данните.
	6. Изчисляване на корелацията по Пиърсън за определяне колко смислена би била препоръка на книга на базата на възрастта на читателя и рейтинга на книгите.
	7. Подготовка на данните и разделянето им на тренировъчни и тестови множества данни.
	8. Създаване и трениране на модела с тренировъчните множества данни.
	9. Грануларно изчисление на разстоянията на текущо изследвания пример и неговите k на брой най-близки съседи.
	10. Препоръка на книга.
	11. Оценка на качеството на модела.

> - **резултати от експеримента**:
> 	1. **Наблюдавани проблеми**: Основен проблем, който срещнахме при този подход е, че трябваше да се установи дали между двата параметъра - години на читателя и рейтинг на книгата съществува линейна връзка, която да ни позволява да използваме регресия за оценка на книгите и потенциални препоръки към читателите. За целта обаче първо трябваше да зачистим данните от незначещи стойности (като NaN стойности). При анализ на данните от трите таблици открихме, че около 30% от читателите не са посочили своята възраст, което доста намали шансовете да постигнем извода за връзка между 	>	въпросните два параметъра, на които искахме да базираме нашите препоръки. Все пак проведохме този експеримент като проверихме коефициента за корелация по метода на Пиърсън.
>	2. **Корелация по Пиърсън**: След като проверихме връзката на рейтинга на книгите и възрастта на читателите, установихме, че p-value-то е равно на 0.0, което от своя страна показва, че между двата компонента не е налична линейна връзка и следователно регресията тук не би довела до добри препоръки на книги.

![recommendations_knn_reg_granular_compute_cor](https://github.com/user-attachments/assets/7ea84789-6bbb-4fe0-b337-11f2c2abb860)

> 	3. **Оценка на качеството на модела**: След като открихме, че този подход не би бил удачен в нашата задача, ние решихме да продължим с него с напълно експериментални цели, за да затвърдим или 	евентуално опровергаем това ни заключение. Завършихме подхода по описаните по-горе стъпки и в последната стъпка при оценката на качеството на модела достигнахме до извода, че едва 40% от 		препоръките са релевантни на предпочитанията на читателите и че около 67% от релевантните препоръки са включени в топ 5 препоръчвани на отделния читател книги, тоест между 1 и 2 от 5 препоръчани на читателя книги наистина съответства на неговите предпочитания. 

![recommendations_knn_reg_granular_compute_eval](https://github.com/user-attachments/assets/b79fcfad-b4ab-4972-a544-827da4e237af)

> - **заключения и бележки за експеримента**: Този резултат само затвърждава по-ранно изведения (чрез корелацията по метода на Пиърсън) извод, че един такъв модел не би бил достатъчно ефективен при препоръките на книги, следователно следва да се мисли в друга насока, например чист k-NN алгоритъм (, който да базира препоръката си само въз основа на рейтинга на книгата) или евентуално оптимизирания k-NN алгоритъм (, познат още като Approximate K Nearest Neighbor - а-NN алгоритъм).

### a-NN алгоритъм
- **HNSW** разновидност
	- Основна идея:
	> Алгоритъмът HNSW (Hierarchical Navigable Small World) е разширена версия на алгоритъма NSW (Navigable Small World), който се базира на концепцията за "малък свят". Тази концепция предполага, че в 	графовете всяка връзка е краткосрочна към съседите (около логаритмично количество стъпки), но също така има и няколко дългосрочни връзки, които позволяват ефективно глобално търсене. Алгоритъмът 	HNSW изгражда многопластова йерархична структура от графове, която позволява ефективно индексиране и търсене на най-близки съседи в високоизмерни пространства.
	- Алгоритъмът HNSW в стъпки:
 - **Faiss** разновидност
	- Основна идея:
	> FAISS ускорява търсенето на най-близки съседи в големи набори от данни, като използва характеристиките на високомерните пространства. Традиционните методи като изчерпателното търсене 		(brute-force) или K-D дърветата са бавни и паметно интензивни при обработка на такива данни. FAISS предлага алгоритми, създадени специално за ефективно търсене в подобни пространства.
	- Алгоритъмът Faiss в стъпки:
 - **ANNOY** разновидност
	- Основна идея:
	> Annoy изгражда йерархична дървовидна структура, наречена Annoy дърво, за да организира данните. Това позволява ефективно търсене на приблизително най-близки съседи. Алгоритъмът жертва малка 	част от точността на резултатите, за да постигне значително по-бързо време за търсене и по-ниска консумация на памет.
	- Алгоритъмът ANNOY в стъпки:
		1. Индексиране (Indexing):
			-  Избира се случайна точка за разделяне по случайно избран размер на данни, която създава кореновия възел на дървото.
				- Данните се разделят рекурсивно в по-малки региони чрез избиране на нови точки за разделяне на различни размери по всяко ниво на дървото.
				- Стремежът е да се разпределят данните равномерно през дървото, с цел балансирана структура на дървото.
		2.  Баланс (Balancing):
			-  Алгоритъмът гарантира, че дървото остава балансирано, като поддържа максимален размер за всеки възел.
				- Ако броят на точките в даден възел надвишава максималния размер, възелът се разделя на два нови дъщерни възела чрез избиране на нова точка за разделяне.
				- Процесът продължава рекурсивно, докато не се създадат всички листни възли в дървото.
		3. Запитвания (Querying):
			-  Алгоритъмът започва от кореновия възел и сравнява точката за запитване с точката за разделяне в текущия възел.
				- Въз основа на това сравнение алгоритъмът избира дъщерния възел, който е по-близо до точката за запитване.
				- Процесът продължава рекурсивно надолу през дървото, като се избира дъщерният възел, който е по-близо до точката за запитване, докато не се достигне листен възел.
		4.  Приближени резултати (Approximate results):
			-  След като се достигне листен възел, алгоритъмът извлича данните в този възел като начален набор от приблизителни най-близки съседи.
				- Алгоритъмът усъвършенства приближените резултати чрез brute-force търсене в ограничен радиус около точката за запитване в листния възел. Радиусът се определя от 					максималното разстояние между точката за запитване и приблизителните най-близки съседи, като алгоритъмът актуализира резултатите с всякакви по-близки точки, открити по 				време на търсенето в радиуса.

## 5.Описание на програмната реализация и примери, илюстриращи работата на програмната 
> По-подробно описание на избрания от нас алгоритъм (след всички тези експерименти), причините зад нашето решение, както и примери, илюстриращи работата на разработената в краен вариант система може да откриете в документацията към проект (**разположена в docs директорията на текущото репо**) и в jupyter notebook-а на проекта (**разположен в src/jupyter директорията на проекта**).

## Конфигурация на проекта
> В тази графа на документацията ще Ви запознаем със стъпките, необходими за успешно конфигуриране и стартиране на изходния код на нашата система на Вашите машини, независимо дали използвате Windows, MacOs или Linux като операционна система.

### Windows / MacOs / Linux
- Стъпка 1: Направете директория, в която искате да разположите кода ни.
- Стъпка 2: Клонирайте следното repository: https://github.com/Nv4n/soz-project-2024-2025.git на вашата машина в току-що създадената от Вас директория. Клонирането може да стане през:
	- терминала със следната команда:  git clone със задаване на директория. Когато клонирате хранилище, можете директно да зададете целева директория, където да бъде изтеглено: 
	**git clone <URL-на-репото> <път-към-директорията>**
	- UI-я на някое IDE (например Visual Studio Code).
- Стъпка 3: Инсталирайте някой Python  Extension Pack и най-необходимите библиотеки, описани по-нагоре в този документ.
- Стъпка 4: Отворете някой от файловете с изходния код, стартирайте го и експериментирайте. 

> Забележка: И за трите операционни системи общите стъпки за конфигурация са еднакви. Разликите могат да идват от това какъв терминал използва самата операционна система или как се извикват конкретни команди.

## Още ресурси
[DATASET](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset)

За информация по [темата](https://towardsdatascience.com/how-did-we-build-bookrecommender-systems-in-an-hour-the-fundamentals-dfee054f978e)
и [темата](https://towardsdatascience.com/how-did-we-build-book-recommender-systems-in-anhour-%20part-2-k-nearest-neighbors-and-matrix-c04b3c2ef55c)

### Starter Kernel(s)

[Recom I: Data Understanding and Simple Recommendation](https://www.kaggle.com/arashnic/recom-i-data-understanding-and-simple-recomm)

### Повече информация

#### My Recommendation Article Series in Medium:

[Evolution of Recommendation Algorithms, Part I: Fundamentals , History Overview, Core and Classical Algorithms](https://medium.com/@anicomanesh/evolution-of-recommendation-algorithms-part-i-fundamentals-and-classical-recommendation-bb1c0bce78a9)
