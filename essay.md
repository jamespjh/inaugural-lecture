# Engineering Computational Science

# Structural note

This document is the 'essay version' of my Inaugural lecture, typed in to help facilitate getting the length and content right, and verify the structure of the argument.

It will be supplemented by some code files for generating the exemplars, and eventually by a for-presentation version in Quarto.

# Part one: An exploration of scientific programming

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

My Dad bought a copy of a book called 'frontiers of complexity' just as I was heading off to uni - I would later have the author of that book as a postdoc supervisor, using computer simulations to model blood flow in the brain. [Wave to Peter.]

[Timeline figure - adding on different points with the icons as we jump back and forth - faded out?]

# 1.1.6 - Effective

My PhD contribution. Rules based approach. Can we automate/computationalise.

# 1.2 Speed - readability tradeoffs

# 1.2.1 Avoiding an unnecessary square root

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

# 1.2.2 Sharing the load

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

# 1.2.3 Warehouse scale computers as toys and tools

We call these 'compute clusters' or 'supercomputers' or 'warehouse scale computers'.

[Illustration - dragonfly network]
[illustration - our datacentre]

Even getting them built is a serious challenge, there's physical complexity in things like getting the heat out of the computer - these things now have fluid pipes that run through them - computer blood - for managing heat! 

In order to get to this point of working with computers-that-share-the-work, there's a lot of complexity that needs to be understood. The interesting part for me is that the complexity of how you structure the machine feeds through into the kind of code you have to write, and vice versa.

# 1.2.4 The abstraction stack

It's a LOT harder to write the program now. And a lot easier to make mistakes. But calculations are possible that would otherwise not be.

Now, you might like it if when you're working in BBC basic - you never need to know about the engineering details of how the memory works. In computer science we call this an "abstraction stack"

[diagram]

... and whenever the concerns of the layer below leak through into the layer above, we call that an "abstraction failure". We're often trying to prevent such failures, by clever forms of 'encapsulating' that complexity. But it never fully works - you can only drive down the rate and signficance - and that's part of why this work is fun and challenging.

This is our second deep lesson: **because in science we use code to understand the world, but the calculations are expensive, the trade off between lucidity and speed really matters**. (We usually say 'complexity and performance'.)

# Part two: Reproducibility and Reliability in Computational Science

# 2.1.1 The Avoiding Mass Extinctions Engine

Why does lucidity in code matter so much? 

Fast forward a few years, to 2009. I'm outside academia for a while, working for a startup doing climate impact modelling - the name of the company was AMEE - the "Avoiding Mass Extinctions Engine". It's a for-profit social enterprise - with a mission to save the world, and make a bob or two doing it. [AMEE bug logo] [Climate matrix symbol]

This really suits me politically - I'm uncomfortable not working for mission/purpose - I want my jobs to be 'for good', but I also like the sense of building a product, of demonstrating that you've got something of real value. This will be important later when we talk about approaches to verifiable understanding.

What our product does is not model the climate - we model polluters. E.g. a computer programme that tells me "if I build a big computer, how much will it harm the climate?" The point here is that Obama was about to pass climate tax legislation, and if that had happened, it would have put such models in the financial-services-business-planning loop.

# 2.1.2 Climategate

In 2009, a group of researchers building computer models of climate at UEA were hacked by climate denialists. In addition to taking various email comments out of context to undermine the science, the hackers exploited bug reports in those emails. Issues with developing computer simulations - part of the work I described in the previous section - were played up to undermine trust in climate science.

As a result of this - a manufactured scandal arising from a hostile disinformation campaign - but with a kernal of truth in the need to improve scientific practice when using software as a critical research instrument - investers were no longer willing to invest in "cleantech" for a while - digital startups working on climate adaptation and mitigation - and AMEE failed.

This really struck a chord with me. There were genuine - innocent - mistakes in the model programming. But there were examples of things where a more careful approach to programming would have prevented these.
And where science is a political football - being used to make decisions that matter for the future of the world - we need to work very carefully - bend over backward - to be very sure our science - and thus the code that drives it - is correct.

By this time I'd realised that effective was also very buggy and badly engineered I should add. My successor in Prof Webber's lab had a lot of tidying up to do!

# 2.1.3 The Joint Biosecurity Centre

<to write>

# 2.2.1 The reproducibility crisis

Given the complexity of the task of effectively using computers in research, it is no wonder we run into these challenges.

We've just seen two examples of where science reaches out from the task of understanding the world to become important in looking after the world. 

But problems with reliability in computationally driven science don't just 

# 2.2.2 Scientism

Science has never followed the image of popperian hypothesis. It's always been a messy social process. 

# 2.2.3 The Research Software Engineer

Growth etc.

# 2.2.4 Open science as a response to reproducibility crisis

* Not just transparency
* Not just efficiency
* Reuse as a driver of correctness
* Why open source means more in science than it does generally

# 2.2.5 FAIR as opposed to Open in an insecure world

# 2.3 What is Engineering?

# 2.3.1 What is it all for?

Boids were done for graphics. Engineering helps science and vice versa.

Reviews of model behaviour, to help build our intuition as to how the system works - not simulations. Push button computer gives answer is not the point.

# 2.3.2 Engineering Computational Science

A definition and a scope

# 2.4 How do we study engineering for computational science?

# 2.4.1 The ESE epistemology

# 2.4.2 The Tinkerer's epistemology

# 2.4.3 The Service Provider's epistemology

leadership lets you get interested in the bits that aren't your main bit - I'm mainly interested in the programming but, e.g. the heat distribution in liquid cooling is fascinating.

# 2.4.4 My appointment and the validation of these epistemologies

# Part three: Some futures

# 3.1 The fourth paradigm. 

The triumph of stamp collecting.

Interpretability and science versus engineering

# 3,2 The fifth paradigm

# 3.2.1 Code as data and model management

# 3.2.2 VIABS and the Beacon Project

# 3.2.3 Models-as-diagrams and the Mathworks



Note on visual representation of code. This will lead through to the pseudocode question.

Pedagogical programming in a world of AI.



Open source, autocomplete and the use of work-that-has-already-been-done.

Pseudocode, DSLs and the chain of compilation/lucidity. In the context of AI assisted engineering.

Abstraction failure of AI prompt as DSL and the role of engineers.