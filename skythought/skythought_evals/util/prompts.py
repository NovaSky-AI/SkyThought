grading_prompt = " \
  You will be given a math problem. Your job is to grade the difficulty level from 1-10 according to the AoPS standard. \
  Here is the standard: \
  {aops_criteria} \
  Problem to be labeled: {problem}. Please put your estimation of the difficulty inside [[level]]. \
  Important: You should place the difficulty from 1-10 into the [[]], not the solution of the problem."

aops_criteria = " \
  All levels are estimated and refer to averages. The following is a rough standard based on the USA tier system AMC 8 - AMC 10 - AMC 12 - AIME - USAMO/USAJMO - IMO, \
  representing Middle School - Junior High - High School - Challenging High School - Olympiad levels. Other contests can be interpolated against this. \
  Notes: \
  Multiple choice tests like AMC are rated as though they are free-response. Test-takers can use the answer choices as hints, and so correctly answer more AMC questions than Mathcounts or AIME problems of similar difficulty. \
  Some Olympiads are taken in 2 sessions, with 2 similarly difficult sets of questions, numbered as one set. For these the first half of the test (questions 1-3) is similar difficulty to the second half (questions 4-6). \
  Scale \
  1: Problems strictly for beginner, on the easiest elementary school or middle school levels (MOEMS, MATHCOUNTS Chapter, AMC 8 1-20, AMC 10 1-10, AMC 12 1-5, and others that involve standard techniques introduced up to the middle school level), most traditional middle/high school word problems. \
  2: For motivated beginners, harder questions from the previous categories (AMC 8 21-25, harder MATHCOUNTS States questions, AMC 10 11-20, AMC 12 5-15, AIME 1-3), traditional middle/high school word problems with extremely complex problem solving. \
  3: Advanced Beginner problems that require more creative thinking (harder MATHCOUNTS National questions, AMC 10 21-25, AMC 12 15-20, AIME 4-6). \
  4: Intermediate-level problems (AMC 12 21-25, AIME 7-9). \
  5: More difficult AIME problems (10-12), simple proof-based Olympiad-style problems (early JBMO questions, easiest USAJMO 1/4). \
  6: High-leveled AIME-styled questions (13-15). Introductory-leveled Olympiad-level questions (harder USAJMO 1/4 and easier USAJMO 2/5, easier USAMO and IMO 1/4). \
  7: Tougher Olympiad-level questions, may require more technical knowledge (harder USAJMO 2/5 and most USAJMO 3/6, extremely hard USAMO and IMO 1/4, easy-medium USAMO and IMO 2/5). \
  8: High-level Olympiad-level questions (medium-hard USAMO and IMO 2/5, easiest USAMO and IMO 3/6). \
  9: Expert Olympiad-level questions (average USAMO and IMO 3/6). \
  10: Historically hard problems, generally unsuitable for very hard competitions (such as the IMO) due to being exceedingly tedious, long, and difficult (e.g. very few students are capable of solving on a worldwide basis). \
  Examples \
  For reference, here are problems from each of the difficulty levels 1-10: \
  <1: Jamie counted the number of edges of a cube, Jimmy counted the numbers of corners, and Judy counted the number of faces. They then added the three numbers. What was the resulting sum? (2003 AMC 8, Problem 1) \
  1: How many integer values of $x$ satisfy $|x| < 3\pi$? (2021 Spring AMC 10B, Problem 1) \
  2: A fair $6$-sided die is repeatedly rolled until an odd number appears. What is the probability that every even number appears at least once before the first occurrence of an odd number? (2021 Spring AMC 10B, Problem 18) \
  3: Triangle $ABC$ with $AB=50$ and $AC=10$ has area $120$. Let $D$ be the midpoint of $\overline{AB}$, and let $E$ be the midpoint of $\overline{AC}$. The angle bisector of $\angle BAC$ intersects $\overline{DE}$ and $\overline{BC}$ at $F$ and $G$, respectively. What is the area of quadrilateral $FDBG$? (2018 AMC 10A, Problem 24) \
  4: Define a sequence recursively by $x_0=5$ and\[x_{n+1}=\frac{x_n^2+5x_n+4}{x_n+6}\]for all nonnegative integers $n.$ Let $m$ be the least positive integer such that\[x_m\leq 4+\frac{1}{2^{20}}.\]In which of the following intervals does $m$ lie? \
  $\textbf{(A) } [9,26] \qquad\textbf{(B) } [27,80] \qquad\textbf{(C) } [81,242]\qquad\textbf{(D) } [243,728] \qquad\textbf{(E) } [729,\infty)$ \
  (2019 AMC 10B, Problem 24 and 2019 AMC 12B, Problem 22) \
  5: Find all triples $(a, b, c)$ of real numbers such that the following system holds:\[a+b+c=\frac{1}{a}+\frac{1}{b}+\frac{1}{c},\]\[a^2+b^2+c^2=\frac{1}{a^2}+\frac{1}{b^2}+\frac{1}{c^2}.\](JBMO 2020/1) \
  6: Let $\triangle ABC$ be an acute triangle with circumcircle $\omega,$ and let $H$ be the intersection of the altitudes of $\triangle ABC.$ Suppose the tangent to the circumcircle of $\triangle HBC$ at $H$ intersects $\omega$ at points $X$ and $Y$ with $HA=3,HX=2,$ and $HY=6.$ The area of $\triangle ABC$ can be written in the form $m\sqrt{n},$ where $m$ and $n$ are positive integers, and $n$ is not divisible by the square of any prime. Find $m+n.$ (2020 AIME I, Problem 15) \
  7: We say that a finite set $\mathcal{S}$ in the plane is balanced if, for any two different points $A$, $B$ in $\mathcal{S}$, there is a point $C$ in $\mathcal{S}$ such that $AC=BC$. We say that $\mathcal{S}$ is centre-free if for any three points $A$, $B$, $C$ in $\mathcal{S}$, there is no point $P$ in $\mathcal{S}$ such that $PA=PB=PC$. \
  Show that for all integers $n\geq 3$, there exists a balanced set consisting of $n$ points. \
  Determine all integers $n\geq 3$ for which there exists a balanced centre-free set consisting of $n$ points. \
  (IMO 2015/1) \
  8: For each positive integer $n$, the Bank of Cape Town issues coins of denomination $\frac1n$. Given a finite collection of such coins (of not necessarily different denominations) with total value at most most $99+\frac{1}{2}$, prove that it is possible to split this collection into $100$ or fewer groups, such that each group has total value at most $1$. (IMO 2014/5) \
  9: Let $k$ be a positive integer and let $S$ be a finite set of odd prime numbers. Prove that there is at most one way (up to rotation and reflection) to place the elements of $S$ around the circle such that the product of any two neighbors is of the form $x^2+x+k$ for some positive integer $x$. (IMO 2022/3) \
  10: Prove that there exists a positive constant $c$ such that the following statement is true: Consider an integer $n > 1$, and a set $\mathcal S$ of $n$ points in the plane such that the distance between any two different points in $\mathcal S$ is at least 1. It follows that there is a line $\ell$ separating $\mathcal S$ such that the distance from any point of $\mathcal S$ to $\ell$ is at least $cn^{-1/3}$. \
  (A line $\ell$ separates a set of points S if some segment joining two points in $\mathcal S$ crosses $\ell$.) (IMO 2020/6)"  # noqa: E501


convert_prompt = "Another solution is written in an unstructured way. Your job is to convert them into two sections: \
    <|begin_of_thought|> \
    (Thought process, you should copy exactly the thinking process of the original solution.) \
    <|end_of_thought|> \
    <|begin_of_solution|> \
    (Final formatted, precise, and clear solution; make sure there is only one solution in this section; If it is a coding problem, make sure there is only one code block) \
    <|end_of_solution|> \
    Here is an example demonstration of a different question, you can refer to its format: \
    {example} \
    Important: You should almost copy all the contents word-by-word of the original solution. Just convert them into two sections. \
    Make sure you include: <|begin_of_slow_thought|>, <|end_of_slow_thought|>,  <|begin_of_solution|>,<|end_of_solution|>  These four headers explicitly.  \
    Content to be converted: {content}"  # noqa: E501

convert_prompt_example = (
    "<|begin_of_thought|>\n\n"
    "Okay, so I've got this problem here. Mr. Wang leaves home at 6 AM, riding his bike at 12 km/h, "
    "and he stops to rest for 6 minutes after every 30 minutes of riding. Then, when he arrives at a park "
    "that's 16.8 km away, I need to find out the angle between the hour and minute hands on his watch.\n\n"
    "Alright, first things first, I need to figure out how long it takes Mr. Wang to ride 16.8 km, including "
    "his rest periods.\n\n"
    "So, his speed is 12 km/h. To find out how long it takes to go 16.8 km without any stops, I can use the formula "
    "time = distance/speed. That would be 16.8 divided by 12, which is 1.4 hours. To make it easier, that's 1 hour and 24 minutes.\n\n"
    "But wait, he doesn't ride straight through. He stops for 6 minutes after every 30 minutes of riding. So, I need to see how many "
    "of those 30-minute riding periods are there in his total riding time.\n\n"
    "In 1 hour and 24 minutes of riding, how many 30-minute segments are there? Well, 1 hour is 60 minutes, plus 24 minutes makes 84 minutes "
    "total riding time. So, 84 divided by 30 is 2.8. That means he has two full 30-minute riding periods and a partial one.\n\n"
    "After each full 30-minute riding period, he rests for 6 minutes. So, for two full periods, he rests twice, which is 12 minutes of rest.\n\n"
    "Now, for the partial riding period. Since 2 times 30 minutes is 60 minutes, and he has 84 minutes of riding, the remaining riding time is 84 minus 60, "
    "which is 24 minutes. So, he rides for 24 minutes without another rest because he doesn't complete another 30-minute segment.\n\n"
    "So, total time taken is riding time plus rest time. That's 84 minutes riding plus 12 minutes resting, totaling 96 minutes.\n\n"
    "Wait a minute, but he stops after every 30 minutes of riding, but in the last partial period of 24 minutes, does he rest again? I think he only rests after "
    "completing 30 minutes of riding, so in this case, since the last riding period is only 24 minutes, he doesn't take an additional rest after that.\n\n"
    "So, total time should be 84 minutes riding plus 12 minutes resting, which is indeed 96 minutes, or 1 hour and 36 minutes.\n\n"
    "So, he leaves at 6 AM and takes 1 hour and 36 minutes to reach the park, arriving at 7:36 AM.\n\n"
    "Now, I need to find the angle between the hour and minute hands at 7:36.\n\n"
    "To find the angle between the hour and minute hands, I can use the formula:\n\n"
    "|30H - 5.5M|\n\n"
    "where H is the hour and M is the minutes.\n\n"
    "At 7:36, H is 7 and M is 36.\n\n"
    "So, plugging in:\n\n"
    "30*7 = 210\n\n"
    "5.5*36 = 198\n\n"
    "210 - 198 = 12\n\n"
    "So, the angle is 12 degrees.\n\n"
    "Wait, but I should make sure that's the smaller angle. Sometimes, the larger angle is considered, but usually, the smaller one is what is asked for.\n\n"
    "So, the angle between the hour and minute hands at 7:36 AM is 12 degrees.\n\n"
    "I think that's the answer.<|end_of_thought|>\n\n"
    "<|begin_of_solution|>\n\n"
    "Mr. Wang leaves home at 6 AM and rides at a speed of 12 km/h, stopping to rest for 6 minutes after every 30 minutes of riding. "
    "He arrives at a park 16.8 km away. To determine the angle between the hour and minute hands on his watch when he arrives, we first calculate the total time taken.\n\n"  # noqa: E501
    "1. **Riding time without stops**:\n\n"
    "$$\\text{Time} = \\frac{\\text{Distance}}{\\text{Speed}} = \\frac{16.8 \\text{ km}}{12 \\text{ km/h}} = 1.4 \\text{ hours} = 84 \\text{ minutes}$$\n\n"
    "2. **Rest periods**:\n\n"
    "  - He rests for 6 minutes after every 30 minutes of riding.\n\n"
    "  - In 84 minutes of riding, he completes 2 full 30-minute segments and a partial 24-minute segment.\n\n"
    "  - He rests twice, totaling 12 minutes of rest.\n\n"
    "3. **Total time**:\n\n"
    "$$\\text{Total time} = 84 \\text{ minutes (riding)} + 12 \\text{ minutes (rest)} = 96 \\text{ minutes} = 1 \\text{ hour and } 36 \\text{ minutes}$$\n\n"
    "  - He arrives at 7:36 AM.\n\n"
    "4. **Angle between hour and minute hands at 7:36**:\n\n"
    "  - Use the formula:\n\n"
    "$$\\text{Angle} = |30H - 5.5M|$$\n\n"
    "  - At 7:36, $H = 7$ and $M = 36$:\n\n"
    "$$\\text{Angle} = |30 \\times 7 - 5.5 \\times 36| = |210 - 198| = 12 \\text{ degrees}$$\n\n"
    "Thus, the angle between the hour and minute hands on his watch is $\\boxed{12}$.<|end_of_solution|>\n"  # noqa: E501
)

# From https://arxiv.org/pdf/2412.09413
system_prompt = "Your role as an assistant involves thoroughly exploring questions through a systematic long \
thinking process before providing the final precise and accurate solutions. This requires \
engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, \
backtracing, and iteration to develop well-considered thinking process. \
Please structure your response into two main sections: Thought and Solution. \
In the Thought section, detail your reasoning process using the specified format: \
<|begin_of_thought|> {thought with steps separated with '\n\n'} \
<|end_of_thought|> \
Each step should include detailed considerations such as analisying questions, summarizing \
relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining \
any errors, and revisiting previous steps. \
In the Solution section, based on various attempts, explorations, and reflections from the Thought \
section, systematically present the final solution that you deem correct. The solution should \
remain a logical, accurate, concise expression style and detail necessary step needed to reach the \
conclusion, formatted as follows: \
<|begin_of_solution|> \
{final formatted, precise, and clear solution} \
<|end_of_solution|> \
Now, try to solve the following question through the above guidelines:"
