# Engineering Computational Science

# Structural note

This document is the 'essay version' of my Inaugural lecture, typed in to help facilitate getting the length and content right, and verify the structure of the argument.

It will be supplemented by some code files for generating the exemplars, and eventually by a for-presentation version in Quarto.

# Part one: An exploration of scientific programming

This lecture has three parts. In the first, which has some practical examples, I attempt to explain some important aspects of what it means to program computers for science.

In the second, I look at the philosophy of computer programming for science, and consider its study as a research discipline. In the third, I look at some possible futures for this field
in the context of the development of deep learning and generative AI.

## 1.1 Pseudocode and the path to scientific understanding

### 1.1.1 A young person's guide to programming gravity

It's the mid-1980s. A BBC micro starts - Beep Boop!. I'm an early teenager, and I've just been reading some stuff about gravity. The explanation sort of makes sense, but I like programming
on my new-ish home computer, and I feel like there's bits of it I haven't understood.

[Picture of BBC micro]
[image - a young person's guide to bbc basic]

I start trying to programme the computer to run a gravity simulation. 

So, kid-me has been told the rules of gravity work like this:

* Accelerate - change your velocity - toward anything heavy
  * How much? Well, the heavier it is, the more you change your speed
    * And it's 'proportional' - so twice as heavy means twice as much change
    * And it goes down with how far away from the 'sun' you are, 
        * proportional to how big the surface of the sphere is between you and the sun (square of radius) 
        * (does this mean the gravity is 'spreading out' so there's the same amount over the sphere?!?)
  * What does it mean 'toward' anything heavy?
    * Well, it's a vector - your velocity is a speed in a particular direction
    * And the change is in the direction of the sun or whatever.

How do we make that in code?

We need to start by examining 'change your velocity'.

We'll work in 2-d, with x and y coordinates. 

Suppose there's no gravity, and we have some speed, and the current speed is vx metres per second in the x direction, and vy metres per second in the y direction.

### 1.1.2 An introduction to pseudocode

Now, this is supposed to be a public lecture about computer programming. How are we going to do this? 

During this lecture, there will be some actual programmable demos I've built, but the code on the screen won't be real code - it'll be text designed to pull out the key aspects I'm trying to explain, that looks a bit like code. (We call it "pseudocode"). Pay attention to this idea of pseudocode - it'll be important!

The demos are all generated using real code - and are available online for those who would like to look at them later - not in BBC basic but in Python so there's some utility for a modern audience.

Then our code looks like:

```
Let's have a time step dt, and make it small, so dt = 0.1
Each time step:
    x = x + vx * dt
    y = y + vy * dt
    Draw it
And that's the end of the time step, so go round and do it again.
```

If we show this animation, we get a dot, moving in a straight line! [Demo]

In order to type this in, we have to have started developing understandings of scalar and vector quantities, and child-me has made, based on the bit of calculus that he's starting to know, some kind of guess at
how small time steps should enable us to implement velocity as change in position - how many metres do you go in a given amount of time.

Note the "loop", a bit  we do the "each time step" bit over and over again, and we indent to show the bit that's looped.

Now we can put in a sun at `(x=0, y=0)`, and add gravity:

```
dt = 0.1
loop:
    distance_to_sun = square_root (x^2 + y^2) -- Pythagoras!
    amount_to_accelerate = gravity_strength_constant * sun_mass / distance_to_sun^2
    x_share_of_acceleration = x / distance_to_sun
    y_share_of_acceleration = y / distance_to_sun
    vx = vx + amount_to_accelerate * x_share_of_acceleration
    vy = vy + amount_to_acceerlate * y_share_of_acceleration
    x = x + vx * dt
    y = y + vy * dt
end of loop
```

We type this in, and watch the video, and it looks like it's working! [Demo]

### 1.1.3 Pseudocode as an intuition pump

The lesson here that I really want to get across here is that for me, and for many of us, the process of building the code, is a really useful way of making sure **there's none of the science you don't understand**.
I definitely didn't understand this properly as a kid, but the point of doing this WASN'T to build a gravity simulation. It was to **use the process of programming as a tool to explore whether you've really understood the thing you're trying to put in the computer**. We call this "modelling" - the computer contains a 'model' of the earth going round the sun. The process of writing the code *even pseudocode* - especially pseudocode - has really forced us to tighten up our thinking.

We don't gain that much more tight thinking from the real code than the pseudocode - but there's a lot more "noise" - this might be the only ACTUAL CODE you see in this lecture:

``` python
def acceleration(position):
    r = np.sqrt(position[0]*position[0] + position[1]*position[1])
    G = 100.0  # Gravitational constant
    g = G / (r * r)  # Gravitational acceleration
    return - g * (position / r) # Make it point toward origin
```

Now, let's think through this a little carefully:

* pseudocode isn't enough - we need a working model to be confident there isn't something we missed
* but once we've gone through that process and it's working, the steps of careful thinking, the rigor that the modelling process has forced us to, isn't in the language syntax
* we're seeing that the process of coding forces us to build scientific intuition - coding is an 'intuition pump'.
* but that the power of this really resides in the pseudocode
* a real programme is necessary for verification we've done the work, but pseudocode is a sufficient embodiment of the intuition

This will be important.

### 1.1.4 Simple isn't always correct

But then we draw the trajectory over time - and instead of being a circle, it doesn't quite meet up! Something is wrong! Oh no, did we misunderstand gravity!?!?! [Figure]

Now, probably quite a few of you in this audience will know that the problem here is the *approach to implementing the calculus* - the small timestep approach we took is called "Forward Euler Integration", and it works quite badly. After asking some annoying questions, this really blew me away as a kid. We followed the 'obvious' approach to implementing the meaning of velocities and accelerations, and it sort of worked, but sort of didn't. A quick trip to Kendal library and an implementation of Runge-Kutte helped sort that out.

So this is another aspect of our lesson about what we learn by programming in science: effectively implementing computer programmes to model the world introduces us to interesting new challenges, in ways that we didn't expect.

### 1.1.5 - Birds and sheep are particles too

Another thing tenage me really liked a few years later - I think I saw this in one of my Dad's books but then did an implentation on Archimedes - is that pretty similar code to this can be used - with a different rule for acceleration - to simulate behaviour of flocking sheep. (Remember I'm from Cumbria.) Or indeed flocking birds - with really beautiful videos. 

You can code in rules for formation flying, avoiding predators etc. [Demo should show the Boids]

The rules for the Boids

[Fill in here]

This also really made an impression on me - the kind of understanding-with-rules, where you aren't properly convinced you've understood something until you can get a computer to do it, works in biology and sociology, and history, as well as physics. 

### 1.1.6 - Effective

My PhD contribution. Rules based approach. Can we automate/computationalise.

## 1.2 Speed - readability tradeoffs

### 1.2.1 Avoiding an unnecessary square root

Some of you will have noticed that the gravity programme is inefficient:

```
    distance_to_sun = square_root (x^2 + y^2) -- Pythagoras!
    amount_to_accelerate = gravity_strength_constant * sun_mass / distance_to_sun^2
```

could be:

```
    amount_to_accelerate = gravity_strength_constant / (x^2 + y^2)
```

...we've saved ourself doing a square root and then a square. Do we want to make that change? The new programme will be faster, but it will be harder to understand. Does that matter?
Well, what if we want to change the programme, or give it to someone else to make changes? My grandad was also into this kind of thing, and it was nice to be able to show him the code.

### 1.2.2 Sharing the load

Let's take this a step further - we're way beyond my teenage example now - but it's another useful bit of the exploration. Suppose we need to simulate lots of gravitating things - maybe a galaxy or something -
and we have to use lots of computers together to solve it.

They need to pass messages between each other, like this:

```
On each computer:
   Loop:
      Do-the-gravity-thing
        And take into account the particles on the other computers, but 'smudged up' rather than one by one.
      For any particles getting to the edge of "my bit":
        Pass them to the computer responsible for the part of the world it's going to.
      Receive from the other computers:
        Any particles getting to me I need to take care of
        And the information about the 'smudged up' mass
```

(In doing this, there's some nice maths about how to do the 'smudging up' about why it's safe, but we're focusing on the computing here.)

[Demo should show a particle being passed over between two simulation domains, and some pretty results for a big system - I might steal something rather than code from scratch depending on time.]

We got a big speed-up from using a supercomputer, but there's LOTS AND LOTS of new complexity. Do our networks work for passing all that information around? How do we do the book keeping of who takes charge of which particle? How do we divide up space between the computers? Does everyone need to talk to everyone? Or can I get away with only talking to my neighbours? Otherwise once I have enough computers, the programme will spend *all it's effort in conversations and no time actually doing gravity maths*. 

This will always happen - anything that doesn't go down as fast as the bit you can properly "parallelise" - i.e. spread across N computers so each takes one Nth of the time - will end up taking all the time! This is called Amdahl's law.

### 1.2.3 Warehouse scale computers as toys and tools

We call these 'compute clusters' or 'supercomputers' or 'warehouse scale computers'.

[Illustration - dragonfly network]
[illustration - our datacentre]

Even getting them built is a serious challenge, there's physical complexity in things like getting the heat out of the computer - these things now have fluid pipes that run through them - computer blood - for managing heat! 

In order to get to this point of working with computers-that-share-the-work, there's a lot of complexity that needs to be understood. The interesting part for me is that the complexity of how you structure the machine feeds through into the kind of code you have to write, and vice versa.

### 1.2.4 The abstraction stack

It's a LOT harder to write the program now. And a lot easier to make mistakes. But calculations are possible that would otherwise not be.

Now, you might like it if when you're working in BBC basic - you never need to know about the engineering details of how the memory works. In computer science we call this an "abstraction stack"

[diagram]

... and whenever the concerns of the layer below leak through into the layer above, we call that an "abstraction failure". We're often trying to prevent such failures, by clever forms of 'encapsulating' that complexity. But it never fully works - you can only drive down the rate and signficance - and that's part of why this work is fun and challenging.

This is our second deep lesson: **because in science we use computers to understand the world, but the calculations are expensive, and the stack is deep, the trade off between lucidity and speed really matters**. (We usually say 'complexity and performance'.)

## 1.3 The Third Paradigm

### 1.3.1 Software as a research tool and the third paradigm. 

### 1.3.2 Complexity science.

My Dad bought a copy of a book called 'frontiers of complexity' just as I was heading off to uni - I would later have the author of that book as a postdoc supervisor, using computer simulations to model blood flow in the brain. [Wave to Peter.]

### 1.3.3 

The model-simulation continuum. Results as a driver of understanding vs the construction process as a driver of understanding.

Reviews of model behaviour, to help build our intuition as to how the system works - not simulations. Push button computer gives answer is not the point.

# Part two: Reproducibility and Reliability in Computational Science

## 2.1 Some examples

### 2.1.1 The Avoiding Mass Extinctions Engine

Fast forward a few years, to 2009. I'm outside academia for a while, working for a startup doing climate impact modelling - the name of the company was AMEE - the "Avoiding Mass Extinctions Engine". It's a for-profit social enterprise - with a mission to save the world, and make a bob or two doing it. [AMEE bug logo] [Climate matrix symbol]

This really suits me politically - I'm uncomfortable not working for mission/purpose - I want my jobs to be 'for good', but I also like the sense of building a product, of demonstrating that you've got something of real value. This will be important later when we talk about approaches to verifiable understanding.

What our product does is not model the climate - we model polluters. E.g. a computer programme that tells me "if I build a big computer, how much will it harm the climate?" The point here is that Obama was about to pass climate tax legislation, and if that had happened, it would have put such models in the financial-services-business-planning loop.

### 2.1.2 Climategate

In 2009, a group of researchers building computer models of climate at UEA were hacked by climate denialists. In addition to taking various email comments out of context to undermine the science, the hackers exploited bug reports in those emails. Issues with developing computer simulations - part of the work I described in the previous section - were played up to undermine trust in climate science.

As a result of this - a manufactured scandal arising from a hostile disinformation campaign - but with a kernal of truth in the need to improve scientific practice when using software as a critical research instrument - investers were no longer willing to invest in "cleantech" for a while - digital startups working on climate adaptation and mitigation - and AMEE failed.

This really struck a chord with me. There were genuine - innocent - mistakes in the model programming. But there were examples of things where a more careful approach to programming would have prevented these.
And where science is a political football - being used to make decisions that matter for the future of the world - we need to work very carefully - bend over backward - to be very sure our science - and thus the code that drives it - is correct.

By this time I'd realised that effective was also very buggy and badly engineered I should add. My successor in Prof Webber's lab had a lot of tidying up to do!

### 2.1.3 The Joint Biosecurity Centre

<to write>

Length of excel file example

### 2.2.1 The reproducibility crisis

Given the complexity of the task of effectively using computers in research, it is no wonder we run into these challenges.

We've just seen examples of where science reaches out from the task of understanding the world to become important in looking after the world, and thus the importance of computational reliability.

But problems with reliability in computationally driven science don't just undermine the correctness of our practical advice or public trust. They undermine the goal of science - to find ways to approach true understandings of the world.

Science, in it's self-conception, views reproducibility - the idea that a paper should enable another scientist to run the same study and verify the conclusion - as an important part of why the so-called scientific method works.

There's growing evidence that across many disciplines this ideal is far from realised.

One might hope that in computational science, where a program constitutes a formal, unambiguous script which describes the model and how it was used, reproducibility would be higher. In fact there reasonable arguments that it's no better, and perhaps worse. [https://pmc.ncbi.nlm.nih.gov/articles/PMC8530091/?utm_source=chatgpt.com]

If you work in computational science you'll have come across loads of these. Genes being automatically renamed by excel to months because they both have three letter codes is one popular example. I've seen - no names today to protect the guilty - published content in a nature paper that was based on numbers where two very large numbers were subtracted from each other - what we call in the trade a 'floating point underflow' - essentially random. It was not retracted.

So we have learned: **we need to build our computer systems for science carefully**, to ensure our science is trustworthy.

## 2.3 What is engineering?

### 2.3.1 A definition

There is a name for the methods humans have come up with to allow us to build things reliably: Engineering.

There are lots of different definitions of engineering around, but one I favour runs something like:

**the tools, practices and systems that enable repeatable, reliable, safe and scalable technical construction**

It's about how we come up with ways to, given our flawed human natures, make things that don't break all the time!

As scientists, we are concerned with understanding the world. 

As engineers, we are concerned not just with wisely using our understanding to make changes to the world that are beneficial, but building ways of working that sustain and protect those benefits.

[Illustration - earthquake]

### 2.3.2 Software Engineering

Software systems now govern our lives, and, as a society, in practice we are *terrible* at it.

[Illustration - post office scandal.]

However, a large body of practice, around testing, documentation, change management and so on has emerged. Applied carefully, these help make it less likely that our software systems will harm us.

It's important for practitioners to know and follow these codes when building code that "matters". I'm a Fellow of The British Computer Society - the UK's professional
institute for information technolgy.

### 2.5.3 Not just *software* engineering

Now, software is not the only aspect of computational science - we've talked about the hardware and network engineering of warehouse-scale computers. 

Computational statistics - these days we call it "data science" - is another aspect of computing with an evolved, but patchily applied system of professional practice. As computing has evolved,
the statistical profession and the computer programming profession have experienced a complex interplay, with rules of practice evolving in response. 

The BCS and the Royal Statistical society work together to build professional practices for data science.

The data itself, is also a key part of this - librarianship is a long-standing professional practice, and the new profession of data stewardship is emerging - a computational librarian of data.

All of these are critical to effective computational science.

### 2.3.3 Ethics and Engineering

A key part of any engineering practice is a sustained ethical framework, and mechanisms for professional practice that ensure that these ethics are upheld. 

I would like to imagine that you haven't reached real professional status as an engineer until you've refused to build something in ways that violate these codes.

I won't go into details in public, but as a data scientist working in the covid response in the UK government, I have achieved this.

[Illustration - phrenology]

### 2.3.4 The Research Software Engineer and the rise of UCL ARC

In 2012 - following some postdocing, and the mathworks, and the climate start up failed, and, needing a job, I came back to academia to do the brain blood flow postdoc with Peter.

With these kind of thoughts around the ethics and practices of engineering and science, I began, with others to think about what kind of intervention might help the reliability and trustworthiness of computational science.

We hypothesised that a professional community of engineers would be necessary for efficient, productive and trustworthy research, given the accelerating importance of computing in science. Today's talk is not mainly about the resulting organisations that we've built together, as I've given that talk a lot before.

Suffice to say, there are now thousands of people around the world employed as part of this set of professions I helped to invent, and a team of a bit less than 150 at UCL, in the UCL Advanced Research Computing Centre, which I lead.

[illustration]

Probably a lot of the reason I'm standing here is the success of this idea - we'll talk more in a moment about whether that's a scientific achievement or not. 

### 2.3.5 The data empowered society

Considered thus, engineering is part of management science, and a social science. 

Sometimes it helps to know some science - but what engineering adds is this layer of tools and approaches, and these are designed in the context of human behaviour and human motivations.

I think this is part of what I bring to my role as co-pro-vice-provost of UCL's Grand Challenge of Data Empowered Societies. 

I'm not a social scientist, but as a practicing engineering leader, I've 
played a significant role in building the social systems that control engineering in computational research. I've been at the sharp end of these issues, and hope that that practical knowledge can help
with the challenge of making the role of data and computers and software in society safe.

## 2.4 Science and Engineering

In the previous section, we talked about professional engineering. I gave examples of long standing and emerging professions in computing - information technology. These professions exist outside the
scientific context - in software and data companies selling cars and T-shirts, in news websites and in the civil service.

I want to dig for a moment into the relationship of science and engineering, touching on four aspects of the relationship.

### 2.4.1 Science-for-engineering

People often think about the question of science-for-engineering: using an understanding of the world, built up from the tools of science, to build useful things to help us manage and control the world.

We've seen examples of this already with climate modelling and covid modelling. It's actually a moderately controversial point as to whether science is that important for engineering. A lot of technologies have been built by trial and error without understanding the science behind them.

[Diagrammatic models of the science-engineering relationship]

Personally, in my work I'm primarily motivated by computational science for science - understanding the world - rather than computational science as a means to technology.

We've seen that in the model-simulation continuum, simulation is of primary utility for engineering, while modelling is of primary utility for science, though this is *not* clear cut.

You'll see that my forays into application are usually driven by a sense of accountability - where the technologies that are related to the science have risks or benefits that matter enough to the world that we have a duty as scientists to be concerned with them.

### 2.4.2 Engineering-for-science

It's also the case that advances in scientific understanding are often driven by engineering goals. Our Boids example - the flocking behaviour - was motivated first by the need to make things that *look like* flocks of birds, for special effects in movies, and then only later applied to animal behaviour as a science!

This is also highly motivating - the role of the application is not only to deliver a technology, but to act as a driver of science. This is a highly signficant feedback loop, and a major part of the real story of how science proceeds.

[Building up a diagram as we go]

### 2.4.3 Engineering studied scientifically

There's a different flavour of science-for-engineering. Not the use of understanding that emerged from science to help with engineering challenges, but rather, the application of the scientific method to discover which tools, practices and systems of engineering, are effective. We can use rigorous methods to investigate questions of which kind of CAD tool is best for building cars, or which structure of an engineering
company delivers most reliably. We'll extensively discuss such approaches in a moment.

### 2.4.4 Engineering applied to science

While all four of these touch my research practice, the fourth category is the one I'm most interested in: the application of engineering to the tools of science? Sometimes this is called "instrument science": 
how best do we build a telescope? A supercollider?

In the first part of the lecture, I tried to convince you that scientific simulation codes, and the computers they run on, are one such interesting engineering challenge in science. How best do we write
trustworthy computer code to be applied as part of the scientific endeavour, and how do we design that alongside the computers it will run on? (We call this "co-design" of software and hardware.)

## 2.5 Engineering Computational Science

### 2.5.1 Definition

So now we're ready to define what I'm interested in - what I claim to be a professor of!

Science is the use of systematic methods to avoid lying to ourselves when trying to understand the world - more on this in a moment.

(When I say the world I mean the human and built worlds as well as the natural world - economics counts.)

Computational science is doing science using information technologies - software, computers and data - as a primary research instrument.

Engineering is the development of tools, practices and systems for the efficient, trustworthy, ethical and safe generation and maintainance of technologies.
Computer engineering is the development and application of those tools and practics to information technology.

Engineering Computational Science is, therefore, the development and application of tools, practices and systems for the efficient,
trustworthy, ethical and safe use of information technologies as part of trying to understand the world.

So now you know what I do.

## 2.6 How do we study engineering for computational science?

### 2.6.1 Epistemic cultures

I remember vividly arriving at university and having my naive, untroubled view of science as a formal, clean way of building understanding of the world absolutely demolished. (This is one of the reasons why it's important to have inter-academic-cultural dialogue across student populations.) I think (except among fundamentalist or propaganda groups who have never practiced in a university), it is quite generally accepted that science is a messy, complex social process.

I do believe - and I think there's ample evidence - that whatever science is, however it works, does work well to help us build understanding. But it's not at all like the simplistic hypothesis-test-theory charicature. We've already seen one example of the richer and more complicated nature of how science moves forward by considering engineering-for-science, science-for-engineering, and engineering science.

For me, there are many different tricks to how we *avoid lying to ourselves*. Sociologists of science call these 'epistemic cultures' - a phrase I learned recently as part of my collaboration with social scientists.

Anyone who is using any of these careful practices to avoid self-deception can reasonably be said, I think, to be doing science. (Some would prefer scholarship, or another term, that's fine,
my only beef is with epistemic exclusionists.)

Over-obsession with a particular epistemology - theory of knowledge - way of building truth, is naive "scientism" - science as a fundamentalist religion, not as science. 

The randomised clinical trial is by far from the only valid research methodology. Don't get me wrong, I love RCTs, and there are colleagues who are making great progress by finding new ways to apply them outside medicine.

But there are rigorous research disciplines that reason from observation, without experiment - like some parts of astronomy. 

There are cultures that learn through deep engagement with small sources - close reading in the humanities for example, or rigourous qualitative research in the social sciences or economics through interview-based methodologies. 

There are whole sub-fields of physics that primarily develop through mathematical reasoning and only occasional access to new data, but make progress from a focus on parsimonious explanation - elegant theories evolving from a heuristic of mathematical beauty. We could spend ages talking about Science vs Wissenschaft.

As we discover new research disciplines, we need to ask, what will be effective self-deception-limiting ways of working, appropriate to the challenge.

### 2.6.2 The ESE epistemology

One powerful approach, but one I do not make use of in my own work - not because I don't believe in it, but just because it doesn't suit my temperament - is the empirical software engineering approach. I need to
explain it carefully now, by way of explaining what my wonderful colleagues who work in this way do, so that I can explain how my approaches differ.

In this approach

### 2.6.3 The Tinkerer's epistemology

### 2.6.4 The Service Provider's epistemology

leadership lets you get interested in the bits that aren't your main bit - I'm mainly interested in the programming but, e.g. the heat distribution in liquid cooling is fascinating.

### 2.6.5 My appointment and the validation of these epistemologies

# Part three: Some futures

## 3.1 The fourth paradigm. 

### 3.1.1 The triumph of stamp collecting.

### 3.1.2 Open science as a response to reproducibility crisis

* Not just transparency
* Not just efficiency
* Reuse as a driver of correctness
* Why open source means more in science than it does generally

### 3.1.3 FAIR as opposed to Open in an insecure world

### 3.1.4 Physics-free gravity 

An exemplar of the fourth paradigm

### 3.1.5 Data science and digital humanities

Humility in discipline crossing

### 3.1.5 Interpretability and fourth paradigm science

## 3.2 The fifth paradigm

### 3.2.1 Code as data and model management

### 3.2.2 VIABS and the Beacon Project

### 3.2.3 Models-as-diagrams and the Mathworks



Note on visual representation of code. This will lead through to the pseudocode question.

Pedagogical programming in a world of AI.



Open source, autocomplete and the use of work-that-has-already-been-done.

Pseudocode, DSLs and the chain of compilation/lucidity. In the context of AI assisted engineering.

Abstraction failure of AI prompt as DSL and the role of engineers.